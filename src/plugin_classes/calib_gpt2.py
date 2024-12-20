import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel
from utils.commons import get_position_ids_text_completion_left_padded, get_position_ids_left_padded


class CustomCalibGPT2ModelBatch(GPT2LMHeadModel):
    def __init__(self, config, ft_model, alpha=0.1):  # added alpha parameter
        super().__init__(config)
        self.config = config
        self.ft_model = ft_model
        self.logits_projector = nn.Linear(self.ft_model.config.vocab_size, config.n_embd)
        self.to_temperature = nn.Linear(config.n_embd, 1)
        self.alpha = alpha  # for the new loss function

    def forward(self, input_ids, attention_mask=None, labels=None, use_new_loss=False):
        # Get base model logits
        with torch.no_grad():
            outputs_base = self.ft_model.forward(input_ids, attention_mask=attention_mask)
            logits_base = outputs_base.logits  # [batch, seq_len, vocab]
        
        # Project logits to hidden size
        hidden_states = self.logits_projector(logits_base)  # [batch, seq_len, n_embd]
        
        # Pass through transformer layer
        transformer_outputs = self.transformer(inputs_embeds=hidden_states, 
                                            attention_mask=attention_mask)
        
        # Get temperature
        temperature = self.to_temperature(transformer_outputs.last_hidden_state)  # [batch, seq_len, 1]
        temperature = temperature.squeeze(-1)  # [batch, seq_len]
        
        # Apply temperature scaling by multiplying with exp(temperature) as per paper
        calibrated_logits = logits_base * torch.exp(temperature).unsqueeze(-1)
        
        if labels is not None:
            shift_logits = calibrated_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            if use_new_loss:
                # Get predicted tokens
                pred_tokens = torch.argmax(shift_logits, dim=-1)  # [batch, seq_len-1]
                
                # Calculate softmax probabilities
                probs = torch.softmax(shift_logits, dim=-1)  # [batch, seq_len-1, vocab]
                
                # Create mask for correct and incorrect predictions
                correct_mask = (pred_tokens == shift_labels)  # [batch, seq_len-1]
                
                # Initialize loss
                loss = torch.zeros_like(correct_mask, dtype=torch.float)
                
                # For correct predictions: standard cross-entropy
                correct_indices = correct_mask.nonzero(as_tuple=True)
                if correct_indices[0].size(0) > 0:
                    loss[correct_indices] = -(1 - self.alpha) * torch.log(
                        probs[correct_indices[0], correct_indices[1], shift_labels[correct_indices]]
                    )
                
                # For incorrect predictions: KL divergence with uniform distribution
                incorrect_indices = (~correct_mask).nonzero(as_tuple=True)
                if incorrect_indices[0].size(0) > 0:
                    vocab_size = shift_logits.size(-1)
                    loss[incorrect_indices] = -self.alpha * torch.mean(
                        torch.log(probs[incorrect_indices[0], incorrect_indices[1], :]), dim=-1
                    )
                
                # Average loss
                loss = loss.mean()
            else:
                # Standard cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return loss, calibrated_logits
        
        return calibrated_logits
