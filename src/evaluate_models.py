import argparse
import evaluate
import logging
import numpy as np
import os
import torch
import yaml
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import pandas as pd

import IPython

from collections import defaultdict
from datasets import load_dataset
# from processed_dataset import DictDataset # somehow not working right now
from nlgmetricverse import NLGMetricverse, load_metric
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils.commons import get_position_ids_left_padded
from utils.data_preprocess import process_e2e_nlg_cleaned, process_common_gen, process_web_nlg, process_nike, process_adidas

def load_adidas_style_words(file_path):
    style_words = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:50]: 
            word, count = line.split(':')
            word = word.strip()
            count = int(count.strip())
            style_words.append((word, count))
    return style_words

def get_token_ids_for_words(words, tokenizer, logger=None):
    token_ids = {}
    for word in words:
        # Try both with and without space prefix
        tokens_no_space = tokenizer(word, add_special_tokens=False)['input_ids']
        tokens_with_space = tokenizer(" " + word, add_special_tokens=False)['input_ids']
        
        # Use the space-prefixed version if it exists as a single token
        if len(tokens_with_space) == 1:
            token_ids[word] = tokens_with_space[0]
        else:
            token_ids[word] = tokens_no_space[0]
            
        # if logger:
        #     logger.info(f"Word '{word}' -> ID {token_ids[word]} -> '{tokenizer.decode([token_ids[word]])}'")
    return token_ids


def plot_style_word_probabilities(plugin_probs, base_probs, style_token_ids, style_words, step, tokenizer, next_token_id, selected_id, save_dir='./probability_plots'):
    plt.figure(figsize=(20, 6))
    
    # Get probabilities and words - using full vocab normalized probabilities directly
    word_probs = []
    for word, _ in style_words:
        space_token = tokenizer(" " + word, add_special_tokens=False)['input_ids']
        if len(space_token) == 1:
            tid = space_token[0]
        else:
            tid = tokenizer(word, add_special_tokens=False)['input_ids'][0]
            
        word_probs.append({
            'word': word,
            'base_prob': base_probs[0, tid].item(),     # Already normalized over full vocab
            'combined_prob': plugin_probs[0, tid].item(),  # Already normalized over full vocab
            'count': next(count for w, count in style_words if w == word)
        })
    
    # Sort by count in descending order
    word_probs.sort(key=lambda x: x['count'], reverse=True)
    
    # Create visualization using full vocab normalized probabilities
    x = np.arange(len(word_probs))
    width = 0.35
    
    plt.bar(x - width/2, [wp['base_prob'] for wp in word_probs], width, 
           label='Base Model', color='lightblue', alpha=0.6)
    plt.bar(x + width/2, [wp['combined_prob'] for wp in word_probs], width, 
           label='Combined', color='lightcoral', alpha=0.6)
    
    plt.xlabel('Style Words (sorted by frequency)')
    plt.ylabel('Probability (normalized over full vocabulary)')
    next_token = tokenizer.decode([next_token_id])
    plt.title(f'Sample {selected_id} - Token Probabilities at Generation Step {step}\nGenerated Token: "{next_token}"')
    
    plt.xticks(x, [f"{wp['word']}" for wp in word_probs], rotation=90, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/sample_{selected_id}_step_{step:03d}_probs.png', dpi=300)
    plt.close()


class DictDataset(Dataset):
    def __init__(self, data_list, tokenizer, input_size):
        """
        Args:
            data_list (list of dicts): A list where each element is a dictionary with features as keys.
        """
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.input_size = input_size

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to retrieve.
        
        Returns:
            dict: A dictionary containing the features and their corresponding values for the given index.
        """
        example = self.data_list[idx]
        
        tokenized = self.tokenizer(example["meaning_representation"], return_tensors="pt", 
                                   max_length=self.input_size, truncation=True, padding="max_length")
        
        
        # Return the tokenized inputs along with any other features (like labels)
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'meaning_representation' :   example['meaning_representation']# Remove batch dimension
        }

def custom_generate_original(model, input_ids, attention_mask, max_length, repetition_penalty, 
                           tokenizer, input_size, top_k=50, temperature=1.0, top_p=1, 
                           bb_model=None, new_model_weight=None, style_token_ids=None, style_words=None, selected_id=None):
    
    # Initialize list to collect all steps' probabilities
    all_steps_data = []
    
    generated_ids = input_ids.clone()  # Start with the input prompt
    finished_sequences = torch.zeros(input_ids.size(0), dtype=torch.bool).to(input_ids.device)
    k = 0
    
    position_ids = get_position_ids_left_padded(input_ids=generated_ids, attention_mask=attention_mask)
        
    # for step in range(max_length-input_ids.size()[1]):
    for step in range(max_length):
        # Get the model outputs (logits) for the current step
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask, use_cache=True, position_ids=position_ids)
            logits = outputs.logits[:, -1, :]  # Get logits of the last token
        k+=1

        # Apply repetition penalty by decreasing the logits for previously generated tokens
        for i, gen_id in enumerate(generated_ids[:, input_size:]):
        # for i, gen_id in enumerate(generated_ids):
            for token_id in torch.unique(gen_id):  # Get unique tokens in the sequence
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= repetition_penalty
                else:
                    logits[i, token_id] *= repetition_penalty

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        if(new_model_weight):
            probs = probs*new_model_weight
        
        if(bb_model):
            with torch.no_grad():
                outputs_base = bb_model(input_ids=generated_ids, attention_mask=attention_mask, use_cache=True, position_ids=position_ids)
                logits_base = outputs_base.logits[:, -1, :]  # Get logits of the last token

            # Apply repetition penalty by decreasing the logits for previously generated tokens
            for i, gen_id in enumerate(generated_ids[:, input_size:]):
            # for i, gen_id in enumerate(generated_ids):
                for token_id in torch.unique(gen_id):  # Get unique tokens in the sequence
                    if logits_base[i, token_id] > 0:
                        logits_base[i, token_id] /= repetition_penalty
                    else:
                        logits_base[i, token_id] *= repetition_penalty
            probs_base = torch.nn.functional.softmax(logits_base, dim=-1)

            if(new_model_weight):
                probs_base = probs_base*(1.0-new_model_weight)

            # Add visualization and data collection here, before probability combination
            if style_token_ids and style_words:
                # Calculate combined probabilities
                if new_model_weight:
                    combined_probs = probs + probs_base
                else:
                    combined_probs = probs * probs_base
                    combined_probs = combined_probs / combined_probs.sum()
                
                # Get actual selected token
                next_token_id = torch.argmax(combined_probs, dim=-1).item()
                
                # Collect data for both visualization and CSV
                word_probs = []
                for word, _ in style_words:
                    space_token = tokenizer(" " + word, add_special_tokens=False)['input_ids']
                    if len(space_token) == 1:
                        tid = space_token[0]
                    else:
                        tid = tokenizer(word, add_special_tokens=False)['input_ids'][0]
                        
                    word_probs.append({
                        'word': word,
                        'base_prob': probs_base[0, tid].item(),
                        'combined_prob': combined_probs[0, tid].item(),
                        'step': step,
                        'next_token': tokenizer.decode([next_token_id])
                    })
                
                # Use same data for plot and CSV
                plot_style_word_probabilities(combined_probs, probs_base, style_token_ids, style_words, 
                                           step, tokenizer, next_token_id, selected_id)
                all_steps_data.extend(word_probs)

            if(new_model_weight):
                probs = probs + probs_base
            else:
                probs = probs*probs_base

            sum_probs = probs.sum(dim=-1, keepdim=True)
    
            # Avoid division by zero by adding a small value (epsilon)
            sum_probs = torch.clamp(sum_probs, min=1e-9)

            # Re-normalize by dividing each probability by the sum of probabilities
            probs = probs / sum_probs

        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
        
        next_token = torch.where(finished_sequences.unsqueeze(-1), tokenizer.pad_token_id, next_token)

        # Append the new token to the generated sequence
        generated_ids = torch.cat((generated_ids, next_token), dim=-1)

        # Extend the attention mask to include the newly generated token
        new_attention_mask = torch.ones((attention_mask.shape[0], 1)).to(input_ids.device)
        attention_mask = torch.cat((attention_mask, new_attention_mask), dim=-1)
        
        finished_sequences |= next_token.squeeze(-1) == tokenizer.eos_token_id
        
        last_values = position_ids[:, -1]  # This gets the last value of each row (shape: m)
        new_values = last_values + 1  # Increment each last value by 1
        new_values = new_values.unsqueeze(1)  # Reshape to (m, 1) to concatenate with the tensor
        position_ids = torch.cat([position_ids, new_values], dim=1) 
        
        if finished_sequences.all():
            # eos_tensor = torch.full( (input_ids.size()[0], (max_length-input_ids.size()[1])-step-1), tokenizer.eos_token_id).to(input_ids.device)
            eos_tensor = torch.full( (input_ids.size()[0], max_length-step-1), tokenizer.eos_token_id).to(input_ids.device)
            generated_ids = torch.cat((generated_ids, eos_tensor), dim=1)
            break

    # Save all steps data to a single CSV at the end
    if all_steps_data:
        df = pd.DataFrame(all_steps_data)
        os.makedirs('./probability_plots', exist_ok=True)
        df.to_csv(f'./probability_plots/sample_{selected_id}_all_steps_probs.csv', index=False)

    return generated_ids[:, input_size:]

def get_eval_dat_per_model(tokenizer, eval_model, dat_loader, device, input_size, meaning_to_references,
                           max_length=128, repetition_penalty=1.1, bb_model = None, new_model_weight=None, style_token_ids=None, style_words=None, selected_id=None, logger=None):
    generated_ids_list = []
    mrs = []
    references = []
    for b, batch in tqdm(enumerate(dat_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        if(bb_model):
            generated_ids = custom_generate_original(eval_model.module, input_ids=input_ids, 
                                                    attention_mask=attention_mask, 
                                                    max_length=max_length, 
                                                    repetition_penalty=repetition_penalty, 
                                                    tokenizer=tokenizer, input_size=input_size,
                                                    new_model_weight = new_model_weight,
                                                    bb_model = bb_model.module,
                                                    style_token_ids=style_token_ids,
                                                    style_words=style_words,
                                                    selected_id=selected_id)
        else:
            generated_ids = custom_generate_original(eval_model.module, input_ids=input_ids, 
                                                    attention_mask=attention_mask, 
                                                    max_length=max_length, 
                                                    repetition_penalty=repetition_penalty, 
                                                    tokenizer=tokenizer, input_size=input_size, 
                                                    new_model_weight = new_model_weight,
                                                    style_token_ids=style_token_ids,
                                                    style_words=style_words,
                                                    selected_id=selected_id)
            
        generated_ids_list.append(generated_ids)
        mrs.extend(batch["meaning_representation"])
        # if(b==0):
        #     break
    generated_ids_tensor = torch.vstack(generated_ids_list)
    predicted_text_list = tokenizer.batch_decode(generated_ids_tensor, skip_special_tokens=True)
    
    for mr in mrs:
        references.append(meaning_to_references[mr])
    
    return {'predictions' : predicted_text_list, 'meaning_representations' : mrs, 'references' : references}

def get_ic_prompt(len_context, dat_val, base_model_name, dataset_name, logger):
    context = ""
    if(dataset_name == 'e2e_nlg_cleaned'):
        if(len_context > 0):
            np.random.seed(42)
            ic_ids = np.random.choice(len(dat_val), len_context)
            context += "Below are examples of (Attributes, Sentence) pairs for some restaurants.\n\n<examples>\n\n"
            if(base_model_name == 'gpt2-medium'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n'
                    )
            elif(base_model_name == 'gpt2-xl'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        '\n' +
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            elif('Llama-3.1-8B' in base_model_name):
                for j, i in enumerate(ic_ids):
                    static_str_from_prompt = 'Please convert the following restaurant attributes into a coherent sentence. Do not provide explanation.\n\n'
                    context += f'Attributes:'
                    context += (
                        dat_val[int(i)]['meaning_representation'][len(static_str_from_prompt):] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            context += "</examples>\n"
    elif(dataset_name == 'web_nlg'):
        if(len_context > 0):
            np.random.seed(42)
            ic_ids = np.random.choice(len(dat_val), len_context)
            if(base_model_name == 'gpt2-medium'):
                context += "Consider the following examples of entity descriptions from facts.\n\n<examples>\n\n"
                for j, i in enumerate(ic_ids):
                    static_str_from_prompt = 'Convert the following facts into a coherent sentence:\n\n'
                    context += f'Example {j+1} --\n'
                    context += (
                        dat_val[int(i)]['meaning_representation'][len(static_str_from_prompt):] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
                context += "</examples>\n"
            elif(base_model_name == 'gpt2-xl'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            elif('Llama-3.1-8B' in base_model_name):
                context += "Consider the following examples of entity descriptions from facts.\n\n<examples>\n\n"
                for j, i in enumerate(ic_ids):
                    static_str_from_prompt = 'Do not provide explanation or follow-up. Just convert the following facts of an entity into a coherent sentence.\n\n'
                    context += f'Example {j+1} --\n'
                    context += (
                        dat_val[int(i)]['meaning_representation'][len(static_str_from_prompt):] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
                context += "</examples>\n"
    elif(dataset_name == 'common_gen'):
        if(len_context > 0):
            np.random.seed(42)
            ic_ids = np.random.choice(len(dat_val), len_context)
            context += "Below are examples of converting given concepts into a coherent sentence.\n\n<start_of_examples>\n\n"
            if(base_model_name == 'gpt2-medium'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            elif(base_model_name == 'gpt2-xl'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        '\n' + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            elif('Llama-3.1-8B' in base_model_name):
                for j, i in enumerate(ic_ids):
                    static_str_from_prompt = 'Please write a coherent sentence that uses all the following concepts.\n\n'
                    # context += f'Example {j+1} --\n'
                    context += (
                        dat_val[int(i)]['meaning_representation'][len(static_str_from_prompt):] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            context += "<end_of_examples>\n"
    elif(dataset_name == 'nike'):
        if(len_context > 0):
            np.random.seed(42)
            ic_ids = np.random.choice(len(dat_val), len_context)
            context += "Below are examples of (Attributes, Advertising description) pairs for some sport products.\n\n<examples>\n\n"
            if(base_model_name == 'gpt2-medium'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n'
                    )
            elif(base_model_name == 'gpt2-xl'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        '\n' +
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            elif('Llama-3.1-8B' in base_model_name):
                for j, i in enumerate(ic_ids):
                    static_str_from_prompt = 'Please write an advertising description of this sport product. Do not provide explanation.\n\n'
                    context += f'Attributes:'
                    context += (
                        dat_val[int(i)]['meaning_representation'][len(static_str_from_prompt):] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            context += "</examples>\n"
    elif(dataset_name == 'adidas'):
        if(len_context > 0):
            np.random.seed(42)
            ic_ids = np.random.choice(len(dat_val), len_context)
            context += "Below are examples of product attributes and their descriptions.\n\n<start_of_examples>\n\n"
            if(base_model_name == 'gpt2-medium'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n'
                    )
            elif(base_model_name == 'gpt2-xl'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        '\n' +
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            elif('Llama-3.1-8B' in base_model_name):
                for j, i in enumerate(ic_ids):
                    # static_str_from_prompt = 'Please write a description of this product. Do not provide explanation.\n\n'
                    static_str_from_prompt = 'Please write a description of this product given the following attributes.\n\n'
                    context += f'Attributes:'
                    context += (
                        dat_val[int(i)]['meaning_representation'][len(static_str_from_prompt):] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            context += "<end_of_examples>\n"
    # logger.info("*************************")
    # logger.info("context: " + context)
    # logger.info("*************************")
    return context

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning base model on the task.")
    parser.add_argument("--model_type", type=str, default='gpt2', help="Plugin and base model type")
    parser.add_argument("--evaluate_model_name", type=str, required=True, help="Moddel to evaluate")
    parser.add_argument("--base_model_name", type=str, default = None, help="Base model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use for evaluation (defaults to 0)")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size of training and evaluation")
    parser.add_argument("--new_model_weight", type=str, default=None, help="Weight for the new model in linear combination")
    parser.add_argument("--len_context", type=int, default=0, help="Number of in-context examples")
    
    args = parser.parse_args()

    if(args.new_model_weight):
        args.new_model_weight = float(args.new_model_weight)

    with open("./configs/evaluate_config.yaml", "r") as file:
        config = yaml.safe_load(file)
        if 'gpt2-medium' in args.evaluate_model_name:
            config['model']['base_model_for_prompt'] = 'gpt2-medium'
        elif 'gpt2-xl' in args.evaluate_model_name:
            config['model']['base_model_for_prompt'] = 'gpt2-xl'
        elif 'Llama-3.1-8B' in args.evaluate_model_name:
            config['model']['base_model_for_prompt'] = 'meta-llama/Llama-3.1-8B'

    os.makedirs(os.path.join(config['logs_dir']), exist_ok = True)

    model_name_list = [
        str(config['data']['dataset_name']), 
        # str(args.evaluate_model_name),
        str(args.evaluate_model_name).replace('/', '_'),
        str(args.batch_size),
        'context',
        str(args.len_context)
    ]
    model_name = '_'.join(model_name_list)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Log level for console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(os.path.join(config['logs_dir'], model_name + '_evaluate.log'))
    file_handler.setLevel(logging.DEBUG)  # Log level for file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # loading tokenizer
    if(args.base_model_name):
        try:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(config['models_dir'], args.base_model_name))
            logger.info('Loaded tokenizer from local base model')
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.base_model_name))
            logger.info('Loaded tokenizer from Huggingface base model')
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(config['models_dir'], args.evaluate_model_name))
            logger.info('Loaded tokenizer from local evaluate model')
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.evaluate_model_name))
            logger.info('Loaded tokenizer from Huggingface evaluate model')
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"tokenizer padding {config['model']['padding_side']}")
    tokenizer.padding_side = config['model']['padding_side']
    # loading base model
    if(args.base_model_name):
        try:
            base_model = AutoModelForCausalLM.from_pretrained(os.path.join(config['models_dir'], args.base_model_name))
            logger.info('Loaded base model from local')
        except OSError:
            if(args.model_type == 'llama'):
                base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, token = config['access_token'], torch_dtype=torch.float16)
                logger.info('Loaded Llama model from Huggingface')
            else:
                base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name)
                logger.info('Loaded GPT model from Huggingface')
    else:
        base_model = None
        logger.info('No base model')
    try:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(config['models_dir'], args.evaluate_model_name))
        logger.info('Loaded evaluate model from local')
    except OSError:
        if(args.model_type == 'llama'):
            model = AutoModelForCausalLM.from_pretrained(args.evaluate_model_name, token = config['access_token'], torch_dtype=torch.float16)
            logger.info('Loaded Llama model for evaluation from Huggingface')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.evaluate_model_name)
            logger.info('Loaded GPT model for evaluation from Huggingface')

    logger.info('Tokenizer, base model, and model loaded')

    # Load and process datasets
    if config['data']['dataset_name'] == 'nike':
        # Nike dataset is processed directly from local CSV
        dataset = process_nike("NikeProductDescriptions.csv", config['model']['base_model_for_prompt'])
        logger.info(f"processed nike data with base model prompt {config['model']['base_model_for_prompt']}")
    elif config['data']['dataset_name'] == 'adidas':
        # Adidas dataset is processed directly from local CSV
        dataset = process_adidas("adidas.csv", config['model']['base_model_for_prompt'])
        logger.info(f"processed adidas data with base model prompt {config['model']['base_model_for_prompt']}")
    elif(config['data']['dataset_name'] == 'web_nlg'):
        # Load and then process web_nlg
        dataset = load_dataset(config['data']['dataset_name'], 'webnlg_challenge_2017', trust_remote_code=True)
        dataset = process_web_nlg(dataset, config['model']['base_model_for_prompt'])
        logger.info(f"processed web_nlg data with base model prompt {config['model']['base_model_for_prompt']}")
    elif(config['data']['dataset_name'] == 'e2e_nlg_cleaned'):
        # Load and then process e2e
        dataset = load_dataset(config['data']['dataset_name'], trust_remote_code=True)
        dataset = process_e2e_nlg_cleaned(dataset, config['model']['base_model_for_prompt'])
        logger.info(f"processed e2e data with base model prompt {config['model']['base_model_for_prompt']}")
    elif(config['data']['dataset_name'] == 'common_gen'):
        # Load and then process common_gen
        dataset = load_dataset(config['data']['dataset_name'], trust_remote_code=True)
        dataset = process_common_gen(dataset, config['model']['base_model_for_prompt'])
        logger.info(f"processed common_gen data with base model prompt {config['model']['base_model_for_prompt']}")
    
    # getting prompt for in-context generation
    context = get_ic_prompt(args.len_context, dataset['validation'], config['model']['base_model_for_prompt'], config['data']['dataset_name'], logger)

    # filtering dataset for evaluation
    dataset = dataset[config['data']['evaluate_tag']]

    meaning_to_references = defaultdict(list)
    for entry in dataset:
        meaning_to_references[context + entry["meaning_representation"]].append(entry["human_reference"])

    logger.info('Dataset filtered based on unique meaning representation')

    
    input_size = config['data']['input_size']
    if(args.len_context > 0):
        input_size += args.len_context*(input_size + config['data']['target_size'])

    # Get all keys and take the last one
    all_mrs = list(meaning_to_references.keys())
    selected_id = 10
    last_mr = all_mrs[selected_id]
    unique_dataset = DictDataset([{'meaning_representation': last_mr}], tokenizer, input_size)

    # Load BLEU and ROUGE metrics from evaluate library
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load('meteor')
    nist_metric = NLGMetricverse(metrics=load_metric("nist"))
    cider_metric = Cider()

    eval_dataloader = DataLoader(unique_dataset, batch_size=args.batch_size, shuffle=False)

    # eval_dataloader = DataLoader(dataset.data[config['data']['evaluate_tag']], 
    #                              batch_size=args.batch_size, shuffle=False)

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # check this line from bash script
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if(base_model):
        base_model = base_model.to(device)
        base_model = torch.nn.DataParallel(base_model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # Load style words and get their token IDs
    style_words = load_adidas_style_words('adidas_style_words.txt')
    style_token_ids = get_token_ids_for_words([word for word, _ in style_words], tokenizer, logger)

    output = get_eval_dat_per_model(tokenizer, model, eval_dataloader, device, 
                                  input_size, meaning_to_references,
                                  max_length=config['data']['target_size'], 
                                  bb_model=base_model, 
                                  new_model_weight=args.new_model_weight,
                                  style_token_ids=style_token_ids,
                                  style_words=style_words,
                                  selected_id=selected_id,
                                  logger=logger)
    
    for sel_id in range(len(output['meaning_representations'])):
        logger.info('Select id:  ' + str(sel_id))
        logger.info('Input:      ' + output['meaning_representations'][sel_id])
        logger.info("--------------------------------")
        logger.info('Prediction: ' + output['predictions'][sel_id])
        logger.info("--------------------------------")
        logger.info('References: ' + ':;:;'.join(output['references'][sel_id]))
        logger.info("--------------------------------")
        # # Add individual BLEU score
        # individual_bleu = bleu_metric.compute(
        #     predictions=[output['predictions'][sel_id]], 
        #     references=[output['references'][sel_id]]
        # )
        # logger.info(f'Individual BLEU: {np.round(individual_bleu["bleu"], 4)}\n')

    # Compute BLEU score
    bleu_score = bleu_metric.compute(predictions=output['predictions'], references=output['references'])
    logger.info(f"BLEU Score: {np.round(bleu_score['bleu'], 4)}")
    # Compute ROUGE score
    rouge_score = rouge_metric.compute(predictions=output['predictions'], references=output['references'])
    for ke in rouge_score.keys():
        logger.info(f"{ke}: {np.round(rouge_score[ke], 4)}")
    # Compute METEOR score
    meteor_score = meteor_metric.compute(predictions=output['predictions'], references=output['references'])
    logger.info(f"METEOR Score: {np.round(meteor_score['meteor'], 4)}")

    # Compute CIDEr score
    cider_score, _ = cider_metric.compute_score(res={k:[v] for k,v in dict(zip(list(range(len(output['predictions']))), output['predictions'])).items()}, 
                                                gts=dict(zip(list(range(len(output['references']))), output['references'])))
    logger.info(f"CIDEr Score: {np.round(cider_score, 4)}")

    # Compute NIST score
    # processing only for nist
    filtered_preds = []
    filtered_refs = []
    for i in range(len(output['predictions'])):
        pred = output['predictions'][i]
        if(len(pred.split(' ')) >= 6): # nist by default assumes 5 n-grams. Taking 6, just to be safe.s
            filtered_preds.append(pred)
            filtered_refs.append(output['references'][i])

    # nist_score = nist_metric.evaluate(predictions=output['predictions'], references=output['references'])
    nist_score = nist_metric.evaluate(predictions=filtered_preds, references=filtered_refs)
    logger.info(f"NIST Score: {np.round(nist_score['nist']['score'], 4)}")


    logger.info('Evaluation Complete.')

if __name__ == "__main__":
    main()