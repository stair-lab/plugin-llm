import argparse
import logging
import os
import torch
import yaml

from callbacks import PrintPredictionsCallback
from processed_dataset import ProcessedDataset
from plugin_classes.plugin_gpt2 import CustomGPT2ModelBatch, GPT2SmallBatch
from plugin_classes.plugin_llama import CustomLlamaModelBatchSeparate
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, 
                          EarlyStoppingCallback, get_linear_schedule_with_warmup, AdamW, GPT2Config, LlamaConfig)
from utils.commons import set_seed
# from callbacks import CustomEarlyStoppingCallback

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import signal
from accelerate import Accelerator  # Add this new import

def signal_handler(signum, frame):
    try:
        logger.info(f"Signal {signum} received. Saving model checkpoint...")
        trainer.save_state()
        trainer.save_model()
        logger.info("Checkpoint saved. Exiting...")
    except Exception as e:
        logger.error(f"Error during checkpoint saving: {str(e)}")
    finally:
        sys.exit(0)

def main():
    # Initialize accelerator before model creation
    accelerator = Accelerator()
    
    # First declare globals
    global trainer, logger, sys
    
    # Then initialize logger
    logger = logging.getLogger(__name__)
    
    # Register signal handler after logger initialization but before training
    if accelerator.is_main_process:
        signal.signal(signal.SIGTERM, signal_handler)
    parser = argparse.ArgumentParser(description="Fine-tuning base model on the task.")
    parser.add_argument("--model_type", type=str, default='gpt2', help="Plugin and base model type")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate of the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of training and evaluation")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay parameter")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to use")
    parser.add_argument("--trained_model_name", type=str, help="Name for the trained model (overrides config)")
    parser.add_argument("--base_model_name", type=str, help="Name of the base model (overrides config)")
    args = parser.parse_args()

    set_seed(args.random_seed)

    with open("./configs/plugin_config.yaml", "r") as file:
        config = yaml.safe_load(file)
        config['model']['trained_model_name'] = args.trained_model_name
        config['model']['base_model_name'] = args.base_model_name

    model_name_list = [
        str(config['model']['trained_model_name']), 
        str(config['data']['dataset_name']), 
        str(config['model']['num_train_epochs']),
        str(args.learning_rate), 
        str(args.batch_size), 
        str(args.weight_decay), 
        str(args.random_seed)
    ]
    model_name_list.append(str(config['model']['base_model_name']).replace('/', '_'))
    model_name = '_'.join(model_name_list)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Log level for console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(os.path.join(config['logs_dir'], model_name + '.log'))
    file_handler.setLevel(logging.DEBUG)  # Log level for file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    # loading tokenizer, base model and creating it

    try:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(config['models_dir'], config['model']['base_model_name']))
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'], token = config['access_token'])
    try:
        base_model = AutoModelForCausalLM.from_pretrained(os.path.join(config['models_dir'], config['model']['base_model_name']))
    except OSError:
        if(args.model_type == 'llama'): # only done for LLama
            base_model = AutoModelForCausalLM.from_pretrained(config['model']['base_model_name'], token = config['access_token'], torch_dtype=torch.float16)
            base_model.to('cuda:3')
        else:
            base_model = AutoModelForCausalLM.from_pretrained(config['model']['base_model_name'], token = config['access_token'])
    # if(config['model']['base_model_name'] == 'gpt2-medium'):
    #     tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
    #     base_model = AutoModelForCausalLM.from_pretrained(config['model']['base_model_name'])
    # else:
    #     model_path = os.path.join(config['models_dir'], config['model']['base_model_name'])
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     base_model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config['model']['padding_side']
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Print model information
    logger.info(f"Base model name: {config['model']['base_model_name']}")
    logger.info(f"Base model parameters: {sum(p.numel() for p in base_model.parameters())}")
    
    # Print base model configuration
    logger.info("Base Model Configuration:")
    logger.info(f"Hidden size: {base_model.config.hidden_size}")
    logger.info(f"Number of attention heads: {base_model.config.num_attention_heads}")
    logger.info(f"Number of layers: {base_model.config.num_hidden_layers}")
    logger.info(f"Vocabulary size: {base_model.config.vocab_size}")
    logger.info(f"Full config: {base_model.config}")
    
    # this is plugin model selection
    if(args.model_type == 'gpt2'):
        if(config['plugin_model']['gpt2']['name']):
            logger.info('Loading pretrained model')
            model = CustomGPT2ModelBatch.from_pretrained(config['plugin_model']['gpt2']['name'], base_model)
        else:
            logger.info('Loading provided config based model')
            gpt2_tmp_config = GPT2Config(**config['plugin_model']['gpt2'])
            logger.info(f"Plugin config: {gpt2_tmp_config}")
            model = CustomGPT2ModelBatch(gpt2_tmp_config, base_model)
    elif(args.model_type == 'llama'):
        if(config['plugin_model']['llama']['name']):
            logger.info('Loading pretrained model')
            model = CustomLlamaModelBatchSeparate.from_pretrained(config['plugin_model']['llama']['name'], tokenizer)
        else:
            logger.info('Loading provided config based model')
            llama_tmp_config = LlamaConfig(**config['plugin_model']['llama'])
            logger.info(f"Plugin config: {llama_tmp_config}")
            model = CustomLlamaModelBatchSeparate(llama_tmp_config, tokenizer)

    # Print plugin model parameters and architecture
    logger.info(f"Plugin model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("Plugin Model Configuration:")
    logger.info(f"Model architecture: {model}")
    
    logger.info('Tokenizer and base model loaded')

    # loading data
    dataset = ProcessedDataset(name=config['data']['dataset_name'], base_model_name=config['model']['base_model_name'])
    
    # processing data
    dataset.mapped_tokenize(tokenizer=tokenizer, input_size=config['data']['input_size'], 
                            target_size=config['data']['target_size'])
    if(config['data']['split_data_name']):
        dataset.split_data(split_key_name=config['data']['split_data_name'], 
                        test_size = config['data']['hyper_train_size'], 
                        random_state = args.random_seed)
    remove_columns = ["meaning_representation", "human_reference"]
    dataset.remove_cols_in_dict(remove_columns)

    for ke in dataset.data.keys():
        logger.info(f'length of {ke} data: {len(dataset.data[ke])}')

    logger.info('Dataset loaded and processed')

    # # Now setting visible devices beyond 0
    # if(args.model_type == 'llama'):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config['results_dir'], model_name),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,  # This batch size is per GPU
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=config['model']['num_train_epochs'],
        # weight_decay=args.weight_decay,
        logging_dir=os.path.join(config['logs_dir'], model_name),
        report_to=["tensorboard"],
        load_best_model_at_end=True,  # Required for early stopping
        metric_for_best_model="eval_loss",  # Metric to determine the best model (optional)
        greater_is_better=False,  # Set to False if lower metric is better (e.g., loss)
        save_total_limit=1,
        seed=args.random_seed,
    )

    if(config['data']['split_data_name']):
        train_dataset = dataset.data[config['data']['split_data_name']]
        eval_dataset = dataset.data[f"hyper{config['data']['split_data_name']}"]
    else:
        train_dataset = dataset.data[config['data']['train_tag']]
        eval_dataset = dataset.data[config['data']['validation_tag']]

    logger.info(f'weight decay is {args.weight_decay}')
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=args.weight_decay)
    for group in optimizer.param_groups:
        logger.info(f"{group['weight_decay']}::{len(group['params'])}")

    total_steps = len(train_dataset) * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['model']['warmup_fraction'] * total_steps),  # Warm-up for the first 10% of steps
        num_training_steps=total_steps
    )

    # Custom Trainer just for logging
    if(args.model_type == 'gpt2'):
        class CustomTrainer(Trainer):
            def log(self, logs):
                # Logs training information to both console and file
                logger.info(logs)
                super().log(logs)
    elif(args.model_type == 'llama'):
        class CustomTrainer(Trainer):
            def log(self, logs):
                # Logs training information to both console and file
                logger.info(logs)
                super().log(logs)

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                # Pass base_model to the custom model's forward method
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'], base_model=base_model)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                return (loss, outputs) if return_outputs else loss

    callbacks=[]  # Add early stopping
    if(config['model']['early_stopping_flag']):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config['model']['early_stopping_patience'], 
                                  early_stopping_threshold=config['model']['early_stopping_threshold']))
        # callbacks.append(EarlyStoppingCallback(patience=config['model']['early_stopping_patience'], 
        #                           threshold=config['model']['early_stopping_threshold']))
    if(config['model']['print_prediction_flag']):
        callbacks.append(PrintPredictionsCallback()) # Does not work with this model at this points
    
    # Initialize the Trainer
    trainer = CustomTrainer(
        model=model,                       # The model with PEFT applied
        args=training_args,                     # Training arguments
        train_dataset=train_dataset, # Training data
        eval_dataset=eval_dataset, # Validation data
        tokenizer = tokenizer,
        optimizers=(optimizer, scheduler),  # Pass optimizer and scheduler
        callbacks=callbacks,
    )

    logger.info('Training the model...')
    # Look for existing checkpoint
    last_checkpoint = None
    checkpoint_dir = os.path.join(config['results_dir'], model_name)
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if 'checkpoint' in f]
        if checkpoints:
            # Sort checkpoints by modification time
            last_checkpoint = os.path.join(
                checkpoint_dir,
                sorted(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))[-1]
            )
            logger.info(f"Found checkpoint: {last_checkpoint}")
            
    # Start or resume training
    trainer.train(resume_from_checkpoint=last_checkpoint)

    model.save_pretrained(os.path.join(config['models_dir'], model_name))
    tokenizer.save_pretrained(os.path.join(config['models_dir'], model_name))

    logger.info('Training complete. Saved the model and tokenizer.')

if __name__ == "__main__":
    main()