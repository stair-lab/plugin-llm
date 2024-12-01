import logging
import torch
from transformers import TrainerCallback, EarlyStoppingCallback

class PrintPredictionsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Log level for console
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)


    def on_evaluate(self, args, state, control, 
                    model=None, tokenizer=None, eval_dataloader=None, num_prints=3, **kwargs):
        # Generate a few predictions
        model.eval()
        for batch in eval_dataloader:
            inputs = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            # Generate predictions
            with torch.no_grad():
                generated_ids = model.generate(inputs, attention_mask=attention_mask, max_length=128 + inputs.size()[1])

            # Filter out invalid token IDs and padding (-100) from inputs and predictions
            def safe_decode(token_ids):
                # Filter out invalid token IDs (e.g., -100) before decoding
                # print(token_ids)
                valid_token_ids = [token_id for token_id in token_ids if 0 <= token_id < tokenizer.vocab_size]
                return tokenizer.decode(valid_token_ids, skip_special_tokens=True)

            # Decode the input, predictions, and true references
            inputs_decoded = [safe_decode(input_ids) for input_ids in inputs]
            preds_decoded = [safe_decode(generated_id) for generated_id in generated_ids]
            refs_decoded = [safe_decode(ref) for ref in batch["labels"]]

            # Print out the input, prediction, and true reference
            for i in range(min(num_prints, len(inputs_decoded))):  # Print up to 3 samples per evaluation
                self.logger.info(f"\nInput (MR): {inputs_decoded[i]}")
                self.logger.info(f"Prediction: {preds_decoded[i]}")
                self.logger.info(f"Reference: {refs_decoded[i]}")
            
            break  # Remove this to print for every batch during evaluation


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# class CustomEarlyStoppingCallback(EarlyStoppingCallback):
#     def __init__(self, early_stopping_patience=3, early_stopping_threshold=0.001):
#         super().__init__(early_stopping_patience, early_stopping_threshold)
#         self.stop_triggered = False  # Flag to detect if early stopping was triggered

#     def on_evaluate(self, args, state, control, **kwargs):
#         # Call the parent method to maintain original behavior
#         control = super().on_evaluate(args, state, control, **kwargs)
        
#         # If early stopping is triggered, set the flag and log a message
#         if control.should_training_stop:
#             self.stop_triggered = True
#             logger.info(f"Early stopping triggered. Training will stop early after {state.epoch:.2f} epochs.")
        
#         return control

#     def on_train_end(self, args, state, control, **kwargs):
#         # If early stopping was triggered, log a final message
#         if self.stop_triggered:
#             logger.info(f"Training stopped early after {state.epoch:.2f} epochs due to early stopping.")
#         super().on_train_end(args, state, control, **kwargs)

# class CustomEarlyStoppingCallback(EarlyStoppingCallback):
#     def on_evaluate(self, args, state, control, **kwargs):
#         # Call the parent method to maintain original behavior
#         super().on_evaluate(args, state, control, **kwargs)
        
#         # If early stopping is triggered, log a message
#         if control.should_training_stop:
#             logger.info(f"Early stopping triggered at epoch {state.epoch:.2f}")

# class CustomEarlyStoppingCallback(EarlyStoppingCallback):
#     def on_evaluate(self, args, state, control, **kwargs):
#         # Call the parent method
#         super().on_evaluate(args, state, control, **kwargs)
        
#         # Log the eval_loss and the improvement after each epoch
#         if state.best_metric is not None:
#             improvement = state.best_metric - state.log_history[-1]['eval_loss']
#             logger.info(f"Epoch {state.epoch}: eval_loss={state.log_history[-1]['eval_loss']}, best_loss={state.best_metric}, improvement={improvement}")
        
#         # If early stopping is triggered, log it
#         if control.should_training_stop:
#             logger.info(f"Early stopping triggered at epoch {state.epoch:.2f}")

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def on_evaluate(self, args, state, control, **kwargs):
        """Called after each evaluation."""
        # Retrieve the last eval_loss from the most recent log entry
        if len(state.log_history) > 0 and 'eval_loss' in state.log_history[-1]:
            eval_loss = state.log_history[-1]['eval_loss']
        else:
            return  # Skip if eval_loss is not available
        
        logger.info(f"$$$$$$$ eval_loss: {eval_loss:.4f}")
        if self.best_loss is not None:
            diff_loss = (self.best_loss - self.threshold)
            logger.info(f"####### diff_loss: {diff_loss:.4f}")

        # Early stopping logic
        if self.best_loss is None:
            self.best_loss = eval_loss  # Set the first eval_loss as the best
            logger.info(f"Initial best eval_loss: {self.best_loss:.4f}")
        elif eval_loss < (self.best_loss - self.threshold):
            self.best_loss = eval_loss  # Update best_loss if significant improvement
            self.counter = 0  # Reset patience counter
            logger.info(f"Epoch {state.epoch}: Improved eval_loss to {eval_loss:.4f}")
        else:
            self.counter += 1  # No significant improvement
            logger.info(f"Epoch {state.epoch}: No improvement in eval_loss. Patience {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {state.epoch}. Best eval_loss: {self.best_loss:.4f}")
                control.should_training_stop = True  # Tell the Trainer to stop training
