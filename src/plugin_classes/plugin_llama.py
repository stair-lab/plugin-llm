import torch
import torch.nn as nn

from transformers import LlamaForCausalLM
from utils.commons import get_position_ids_left_padded, get_base_probs

class CustomLlamaModelBatchSeparate(LlamaForCausalLM):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.config = config
        super().resize_token_embeddings(len(tokenizer))
    
    def forward(self, input_ids, attention_mask=None, labels=None, base_model=None):
        # Get the logits from the base GPT-2 model
        position_ids = get_position_ids_left_padded(input_ids=input_ids, 
                                                    attention_mask=attention_mask)
        outputs = super().forward(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        
        probabilities_base = get_base_probs(base_model, input_ids, attention_mask, position_ids)
        
        probabilities_base = probabilities_base.to(probabilities.device)
        # print('no error here as well')
        
        # Multiply the probabilities by the custom vector
        # Ensure that the custom vector shape matches the logits shape [batch_size, seq_len, vocab_size]
        # For this example, assume the vector applies element-wise across the vocabulary dimension
        weighted_probabilities = probabilities * probabilities_base
        # weighted_probabilities = probabilities
        
        # Normalize the probabilities again to ensure they sum to 1
        normalized_probabilities = weighted_probabilities / weighted_probabilities.sum(dim=-1, keepdim=True)
        
        # Convert back to logits (unnormalized scores) by applying log after multiplying probabilities
        modified_logits = torch.log(normalized_probabilities + 1e-8)  # Add small constant to avoid log(0)
        
        # Use modified logits for loss calculation or generation
        if labels is not None:
            # Shift the logits and labels for computing the loss
            shift_logits = modified_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, modified_logits
        
        return modified_logits