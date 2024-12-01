import torch
import torch.nn as nn

from transformers import LlamaForCausalLM
from utils.commons import get_position_ids_left_padded, get_base_probs

class CustomLlamaCombinedModelBatchSeparate(LlamaForCausalLM):
    def __init__(self, config, tokenizer, new_model_weight):
        super().__init__(config)
        self.config = config
        super().resize_token_embeddings(len(tokenizer))
        self.new_model_weight = new_model_weight
    
    def forward(self, input_ids, attention_mask=None, labels=None, base_model=None):
        position_ids = get_position_ids_left_padded(input_ids=input_ids, attention_mask=attention_mask)
        outputs = super().forward(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)

        probabilities = probabilities*self.new_model_weight
        
        if(self.new_model_weight<1):
            probabilities_base = get_base_probs(base_model, input_ids, attention_mask, position_ids)
            probabilities_base = probabilities_base*(1.0 - self.new_model_weight)
        else:
            probabilities_base = torch.zeros_like(probabilities)

        probabilities_base = probabilities_base.to(probabilities.device)
        
        # weighted_probabilities = probabilities * probabilities_base
        comb_probabilities = probabilities + probabilities_base
        
        # normalized_probabilities = weighted_probabilities / weighted_probabilities.sum(dim=-1, keepdim=True)
        # modified_logits = torch.log(normalized_probabilities + 1e-8)  # Add small constant to avoid log(0)
        modified_logits = torch.log(comb_probabilities + 1e-8)  # Add small constant to avoid log(0)

        
        if labels is not None:
            shift_logits = modified_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, modified_logits
        
        return modified_logits