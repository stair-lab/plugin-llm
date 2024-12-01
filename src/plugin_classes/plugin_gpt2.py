import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel
from utils.commons import get_position_ids_text_completion_left_padded, get_position_ids_left_padded


class CustomGPT2ModelBatch(GPT2LMHeadModel):
    def __init__(self, config, ft_model):
        super().__init__(config)
        self.config = config
        self.ft_model = ft_model

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

        with torch.no_grad():
            outputs_base = self.ft_model.forward(input_ids, attention_mask=attention_mask, 
                                                 position_ids=position_ids)
            logits_base = outputs_base.logits
            probabilities_base = nn.functional.softmax(logits_base, dim=-1)
        
        weighted_probabilities = probabilities * probabilities_base
        
        normalized_probabilities = weighted_probabilities / weighted_probabilities.sum(dim=-1, keepdim=True)
        
        modified_logits = torch.log(normalized_probabilities + 1e-8)  # Add small constant to avoid log(0)
        
        # Use modified logits for loss calculation or generation
        if labels is not None:
            shift_logits = modified_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, modified_logits
        
        return modified_logits
    
class GPT2SmallBatch(GPT2LMHeadModel):
    def __init__(self, config, ft_model):
        super().__init__(config)
        self.ft_model = ft_model
        for param in super().parameters():
            param.requires_grad = False
        super().transformer.wte.weight.requires_grad = True


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

        with torch.no_grad():
            outputs_base = self.ft_model.forward(input_ids, attention_mask=attention_mask, 
                                                 position_ids=position_ids)
            logits_base = outputs_base.logits
            probabilities_base = nn.functional.softmax(logits_base, dim=-1)
        
        weighted_probabilities = probabilities * probabilities_base
        
        normalized_probabilities = weighted_probabilities / weighted_probabilities.sum(dim=-1, keepdim=True)
        
        modified_logits = torch.log(normalized_probabilities + 1e-8)  # Add small constant to avoid log(0)
        
        # Use modified logits for loss calculation or generation
        if labels is not None:
            shift_logits = modified_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, modified_logits
        
        return modified_logits
    

class CustomGPT2Model(GPT2LMHeadModel):
    def __init__(self, config, ft_model, target_size=128):
        super().__init__(config)
        self.config = config
        self.ft_model = ft_model
        self.target_size = target_size
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        
        generated_ids = input_ids.clone()  # Start with the input prompt
        modified_logits = torch.empty(input_ids.size(0), self.target_size, 
                                      self.ft_model.config.vocab_size).to(input_ids.device)  # Empty tensor of the desired final size

        with torch.no_grad():
            position_ids = get_position_ids_left_padded(input_ids=generated_ids, attention_mask=attention_mask)
 
        for step in range(self.target_size):
            outputs = super().forward(input_ids=generated_ids, 
                                      attention_mask=attention_mask, 
                                      position_ids=position_ids)
            logits = outputs.logits[:, -1, :]  # Get logits of the last token
            probs = torch.nn.functional.softmax(logits, dim=-1)

            with torch.no_grad():
                outputs_base = self.ft_model.forward(input_ids=generated_ids, 
                                                     attention_mask=attention_mask, 
                                                     position_ids=position_ids)
                logits_base = outputs_base.logits[:, -1, :]  # Get logits of the last token
                probs_base = torch.nn.functional.softmax(logits_base, dim=-1)
                    
            probs = probs*probs_base
            sum_probs = probs.sum(dim=-1, keepdim=True)

            # Avoid division by zero by adding a small value (epsilon)
            sum_probs = torch.clamp(sum_probs, min=1e-9)

            # Re-normalize by dividing each probability by the sum of probabilities
            probs = probs / sum_probs
            
            next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
            
            generated_ids = torch.cat((generated_ids, next_token), dim=-1)
            # Extend the attention mask to include the newly generated token
            new_attention_mask = torch.ones((attention_mask.shape[0], 1)).to(input_ids.device)
            attention_mask = torch.cat((attention_mask, new_attention_mask), dim=-1)
            
            temp_logits = torch.log(probs + 1e-8) 
            
            # modified_logits = torch.cat((modified_logits, temp_logits), dim=1)
            modified_logits[:, step, :] = temp_logits
            
            last_values = position_ids[:, -1]  # This gets the last value of each row (shape: m)
            new_values = last_values + 1  # Increment each last value by 1
            new_values = new_values.unsqueeze(1)  # Reshape to (m, 1) to concatenate with the tensor
            position_ids = torch.cat([position_ids, new_values], dim=1) 
            

        if labels is not None:            
            shift_logits = modified_logits
            shift_labels = labels
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, modified_logits
        
        return modified_logits