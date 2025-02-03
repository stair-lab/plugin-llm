import argparse
import logging
import os
import yaml

from callbacks import PrintPredictionsCallback
from processed_dataset import ProcessedDataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, 
                          EarlyStoppingCallback, get_linear_schedule_with_warmup, AdamW)
from utils.commons import set_seed


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning base model on the task.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate of the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of training and evaluation")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay parameter")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to use")
    args = parser.parse_args()

    set_seed(args.random_seed)

    with open("./configs/base_model_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_name_list = [
        str(config['model']['trained_model_name']), 
        str(config['data']['dataset_name']),
        str(config['model']['base_model_name']),
        str(config['model']['num_train_epochs']),
        str(args.learning_rate), 
        str(args.batch_size), 
        str(args.weight_decay), 
        str(args.random_seed)
    ]
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

    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config['model']['padding_side']
    # loading base model
    model = AutoModelForCausalLM.from_pretrained(config['model']['base_model_name'])

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

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config['results_dir'], model_name),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,  # This batch size is per GPU
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=config['model']['num_train_epochs'],
        weight_decay=args.weight_decay,
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

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    total_steps = len(train_dataset) * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['model']['warmup_fraction'] * total_steps),  # Warm-up for the first 10% of steps
        num_training_steps=total_steps
    )

    # Custom Trainer just for logging
    class CustomTrainer(Trainer):
        def log(self, logs):
            # Logs training information to both console and file
            logger.info(logs)
            super().log(logs)

    callbacks=[]  # Add early stopping
    if(config['model']['early_stopping_flag']):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config['model']['early_stopping_patience'], 
                                  early_stopping_threshold=config['model']['early_stopping_threshold']))
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
    trainer.train()

    model.save_pretrained(os.path.join(config['models_dir'], model_name))
    tokenizer.save_pretrained(os.path.join(config['models_dir'], model_name))

    logger.info('Training complete. Saved the model and tokenizer.')

if __name__ == "__main__":
    main()