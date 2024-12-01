import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel
from utils.commons import get_position_ids_text_completion_left_padded, get_position_ids_left_padded


class CustomGPT2CombinedModelBatch(GPT2LMHeadModel):
    def __init__(self, config, ft_model, new_model_weight):
        super().__init__(config)
        self.config = config
        self.ft_model = ft_model
        self.new_model_weight = new_model_weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get the logits from the base GPT-2 model
        # this is useful for padding, mr, padding, hr
        # position_ids = get_position_ids_text_completion_left_padded(input_ids=input_ids, 
        #                                                             attention_mask=attention_mask)
        # this is useful for padding, mr, hr
        position_ids = get_position_ids_left_padded(input_ids=input_ids, 
                                                    attention_mask=attention_mask)
        outputs = super().forward(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)

        probabilities = probabilities*self.new_model_weight

        if(self.new_model_weight<1):
            with torch.no_grad():
                outputs_base = self.ft_model.forward(input_ids, attention_mask=attention_mask, 
                                                    position_ids=position_ids)
                logits_base = outputs_base.logits
                probabilities_base = nn.functional.softmax(logits_base, dim=-1)
                probabilities_base = probabilities_base*(1.0-self.new_model_weight)
        else:
            probabilities_base = torch.zeros_like(probabilities)
            probabilities_base = probabilities_base.to(probabilities.device)
        
        comb_probabilities = probabilities + probabilities_base
        
        # normalized_probabilities = weighted_probabilities / weighted_probabilities.sum(dim=-1, keepdim=True)
        # modified_logits = torch.log(normalized_probabilities + 1e-8)  # Add small constant to avoid log(0)
        
        modified_logits = torch.log(comb_probabilities + 1e-8)
        
        # Use modified logits for loss calculation or generation
        if labels is not None:
            shift_logits = modified_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, modified_logits
        
        return modified_logits