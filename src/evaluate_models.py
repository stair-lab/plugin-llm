import argparse
import evaluate
import logging
import numpy as np
import os
import torch
import yaml

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
from utils.data_preprocess import process_e2e_nlg_cleaned, process_common_gen, process_web_nlg


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
        
        # # Tokenize the 'meaning_representation' on the fly
        # tokenized = tokenize_function(example, self.tokenizer)
        # inputs = f'<bos> {example["meaning_representation"]} <eos>'
        # inputs = f"Question: Generate a natural language sentence from the following aspects: {example['meaning_representation']}" + "\nAnswer:"
        # if(self.base_model_name):
        #     if('gpt2-medium' in self.base_model_name):
        #         prefix_str = 'Given the following aspects of a restaurant, "'
        #         suffix_str = '", a natural language sentence describing the restuarant is: '
        #     elif('Llama-3.1-8B' in self.base_model_name):
        #         prefix_str = 'Question: Given the following attributes of a restaurant, "'
        #         suffix_str = '", how would you describe the restaurant based on the attributes? Just provide the description with no explanation.\nAnswer: '
        #     else:
        #         prefix_str = ''
        #         suffix_str = ''
        # else:
        #     if(self.model_type == 'gpt2'):
        #         prefix_str = 'Given the following aspects of a restaurant, "'
        #         suffix_str = '", a natural language sentence describing the restuarant is: '
        #     elif(self.model_type == 'llama'):
        #         prefix_str = 'Question: Given the following attributes of a restaurant, "'
        #         suffix_str = '", how would you describe the restaurant based on the attributes? Just provide the description with no explanation.\nAnswer: '
        #     else:
        #         prefix_str = ''
        #         suffix_str = ''
        # inputs = prefix_str + example["meaning_representation"] + suffix_str
        # inputs = f'{example["meaning_representation"]}'
        tokenized = self.tokenizer(example["meaning_representation"], return_tensors="pt", 
                                   max_length=self.input_size, truncation=True, padding="max_length")
        
        
        # Return the tokenized inputs along with any other features (like labels)
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'meaning_representation' :   example['meaning_representation']# Remove batch dimension
        }


def custom_generate_original(model, input_ids, attention_mask, max_length, repetition_penalty, 
                             tokenizer, input_size, top_k=50, temperature=1.0, top_p = 1, bb_model = None, new_model_weight = None):
    
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

    return generated_ids[:, input_size:]

def get_eval_dat_per_model(tokenizer, eval_model, dat_loader, device, input_size, meaning_to_references,
                           max_length=128, repetition_penalty=1.1, bb_model = None, new_model_weight=None):
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
                                                    bb_model = bb_model.module)
        else:
            generated_ids = custom_generate_original(eval_model.module, input_ids=input_ids, 
                                                    attention_mask=attention_mask, 
                                                    max_length=max_length, 
                                                    repetition_penalty=repetition_penalty, 
                                                    tokenizer=tokenizer, input_size=input_size, 
                                                    new_model_weight = new_model_weight)
            
        generated_ids_list.append(generated_ids)
        mrs.extend(batch["meaning_representation"])
        # if(b==0):
        #     break
    generated_ids_tensor = torch.vstack(generated_ids_list)
    predicted_text_list = tokenizer.batch_decode(generated_ids_tensor, skip_special_tokens=True)
    for mr in mrs:
        references.append(meaning_to_references[mr])
    
    return {'predictions' : predicted_text_list, 'meaning_representations' : mrs, 'references' : references}

def get_ic_prompt(len_context, dat_val, base_model_name, dataset_name):
    context = ""
    if(dataset_name == 'e2e_nlg_cleaned'):
        if(len_context > 0):
            np.random.seed(42)
            ic_ids = np.random.choice(len(dat_val), len_context)
            if(base_model_name == 'gpt2-medium'):
                context += "<examples>\n"
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n'
                    )
                context += "</examples>\n"
            elif(base_model_name == 'gpt2-xl'):
                for j, i in enumerate(ic_ids):
                    context += (
                        dat_val[int(i)]['meaning_representation'] + 
                        '\n' + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
            elif('Llama-3.1-8B' in base_model_name):
                context += "Consider the following examples of restaurant descriptions from attributes.\n\n<examples>\n\n"
                for j, i in enumerate(ic_ids):
                    static_str_from_prompt = 'Do not provide explanation. Just convert the following attributes of a restaurant in a coherent sentence.\n\n'
                    context += f'Example {j+1} --\n'
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
                    static_str_from_prompt = 'Do not provide explanation. Just generate a single coherent sentence based on the following concepts.\n\n'
                    # context += f'Example {j+1} --\n'
                    context += (
                        dat_val[int(i)]['meaning_representation'][len(static_str_from_prompt):] + 
                        dat_val[int(i)]['human_reference'] + 
                        '\n\n'
                    )
                context += 'Consider the above coherent sentences from concepts. Just generate one sentence. '
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

    with open("../configs/evaluate_config.yaml", "r") as file:
        config = yaml.safe_load(file)

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

    # dataset = load_dataset(config['data']['dataset_name'], split=config['data']['evaluate_tag'])
    if(config['data']['dataset_name'] == 'web_nlg'):
        dataset = load_dataset(config['data']['dataset_name'], 'webnlg_challenge_2017', trust_remote_code=True)
    else:
        dataset = load_dataset(config['data']['dataset_name'], trust_remote_code=True)

    if(config['data']['dataset_name'] == 'e2e_nlg_cleaned'):
        dataset = process_e2e_nlg_cleaned(dataset, config['model']['base_model_for_prompt'])
        logger.info(f"processed e2e data with base model prompt {config['model']['base_model_for_prompt']}")
    elif(config['data']['dataset_name'] == 'web_nlg'):
        dataset = process_web_nlg(dataset, config['model']['base_model_for_prompt'])
        logger.info(f"processed web_nlg data with base model prompt {config['model']['base_model_for_prompt']}")
    elif(config['data']['dataset_name'] == 'common_gen'):
        dataset = process_common_gen(dataset, config['model']['base_model_for_prompt'])
        logger.info(f"processed common_gen data with base model prompt {config['model']['base_model_for_prompt']}")
    
    # getting prompt for in-context generation
    context = get_ic_prompt(args.len_context, dataset['validation'], config['model']['base_model_for_prompt'], config['data']['dataset_name'])

    # filtering dataset for evaluation
    dataset = dataset[config['data']['evaluate_tag']]


    # # loading data
    # dataset = ProcessedDataset(name=config['data']['dataset_name'])
    # # processing data
    # dataset.mapped_tokenize_for_evaluate(tokenizer=tokenizer, input_size=config['data']['input_size'])
    # print('$$$$', type(dataset.data[config['data']['evaluate_tag']][0]['input_ids']))
    # # remove_columns = ["meaning_representation", "human_reference"]
    # # remove_columns = ["human_reference"]
    # # dataset.remove_cols_in_dict(remove_columns)
    # logger.info('Dataset loaded and processed')

    meaning_to_references = defaultdict(list)
    # for entry in dataset[config['data']['evaluate_tag']]:
    for entry in dataset:
        meaning_to_references[context + entry["meaning_representation"]].append(entry["human_reference"])
    # Getting only unique meaning_representations

    # dataset.filter_rows_in_list('meaning_representation', list(meaning_to_references.keys()))

    # dataset.map_convert_to_tensors()
    # print('####', type(dataset.data[config['data']['evaluate_tag']][0]['input_ids']))
    # print('####', dataset.data[config['data']['evaluate_tag']][0])

    # for ke in dataset.data.keys():
    #     logger.info(f'length of {ke} data: {len(dataset.data[ke])}')

    logger.info('Dataset filtered based on unique meaning representation')

    
    input_size = config['data']['input_size']
    if(args.len_context > 0):
        input_size += args.len_context*(input_size + config['data']['target_size'])

    unique_dataset = DictDataset([{'meaning_representation': mr} for mr in meaning_to_references.keys()], 
                                 tokenizer, input_size)

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

    output = get_eval_dat_per_model(tokenizer, model, eval_dataloader, device, 
                                    input_size, meaning_to_references,
                                    max_length=config['data']['target_size'], bb_model = base_model, new_model_weight=args.new_model_weight)
    
    for sel_id in range(len(output['meaning_representations'])):
        logger.info('Select id:  ' + str(sel_id))
        logger.info('Input:      ' + output['meaning_representations'][sel_id])
        logger.info('Prediction: ' + output['predictions'][sel_id])
        logger.info('References: ' + ':;:;'.join(output['references'][sel_id]))

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


    # # Display the BLEU, ROUGE, and METEOR scores
    # logger.info(f"BLEU Score: {np.round(bleu_score['bleu'], 4)}")
    # for ke in rouge_score.keys():
    #     logger.info(f"{ke}: {np.round(rouge_score[ke], 4)}")
    # logger.info(f"METEOR Score: {np.round(meteor_score['meteor'], 4)}")
    # logger.info(f"NIST Score: {np.round(nist_score['nist']['score'], 4)}")
    # logger.info(f"CIDEr Score: {np.round(cider_score, 4)}")

    logger.info('Evaluation Complete.')

if __name__ == "__main__":
    main()