import numpy as np
import random
import torch

def get_position_ids_left_padded(input_ids, attention_mask):
    """
    Generate position IDs for left-padded sequences.
    
    Args:
        input_ids (torch.Tensor): Tensor of input token IDs (shape: [batch_size, seq_len]).
        attention_mask (torch.Tensor): Tensor of attention mask (shape: [batch_size, seq_len]).
        
    Returns:
        torch.Tensor: Position IDs (shape: [batch_size, seq_len]).
    """
    # Get the lengths of the non-padded tokens (i.e., count of '1's in the attention mask)
    seq_lengths = attention_mask.sum(dim=-1)

    # Create a tensor with position IDs starting from 0 for each non-padded token
    position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)

    # Adjust position IDs for each sequence to start from 0 after padding
    position_ids = position_ids - (input_ids.size(1) - seq_lengths).unsqueeze(-1)

    # Set position IDs for padding tokens to 0 (optional: you can use another value if needed)
    position_ids = torch.where(attention_mask == 1, position_ids, torch.zeros_like(position_ids))
    return position_ids.long()

def get_position_ids_text_completion_left_padded(input_ids, attention_mask, split_point=64):
    """
    Generate position IDs for left-padded sequences, where input_ids contains both the 
    meaning representation (MR) and human reference (target), concatenated together.
    
    Args:
        input_ids (torch.Tensor): Tensor of input token IDs (shape: [batch_size, seq_len]).
        attention_mask (torch.Tensor): Tensor of attention mask (shape: [batch_size, seq_len]).
        split_point (int): The index at which the meaning representation ends and the human reference begins.
        
    Returns:
        torch.Tensor: Position IDs (shape: [batch_size, seq_len]).
    """
    # Step 1: Process the meaning representation (MR) part of the input_ids
    # Create a tensor with position IDs starting from 0 for each non-padded token in the MR
    mr_attention_mask = attention_mask[:, :split_point]
    mr_position_ids = torch.arange(split_point, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)
    
    # Get the lengths of the non-padded tokens in the MR
    mr_seq_lengths = mr_attention_mask.sum(dim=-1)

    # Adjust position IDs for MR to start from 0 after padding
    mr_position_ids = mr_position_ids - (split_point - mr_seq_lengths).unsqueeze(-1)
    
    # Set position IDs for padding tokens in MR to 0
    mr_position_ids = torch.where(mr_attention_mask == 1, mr_position_ids, torch.zeros_like(mr_position_ids))

    # Step 2: Process the human reference (target) part of the input_ids
    target_attention_mask = attention_mask[:, split_point:]
    target_seq_len = input_ids.size(1) - split_point
    target_position_ids = torch.arange(target_seq_len, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)
    
    # Get the lengths of the non-padded tokens in the human reference
    target_seq_lengths = target_attention_mask.sum(dim=-1)

    # Adjust position IDs for human reference to start from 0 after padding
    target_position_ids = target_position_ids - (target_seq_len - target_seq_lengths).unsqueeze(-1)
    
    target_position_ids += mr_seq_lengths.unsqueeze(1).repeat(1, target_position_ids.size()[1])
    
    # Set position IDs for padding tokens in human reference to 0
    target_position_ids = torch.where(target_attention_mask == 1, target_position_ids, torch.zeros_like(target_position_ids))

    # Step 3: Concatenate the position IDs for MR and human reference
    position_ids = torch.cat([mr_position_ids, target_position_ids], dim=-1)
    
    return position_ids.long()

def set_seed(seed):
    """Set the seed for reproducibility."""
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    
    # Ensure deterministic behavior by setting PyTorch to deterministic mode (optional)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_base_probs(md, input_ids, attention_mask, position_ids):
    with torch.no_grad():
        position_ids_base = position_ids.to(md.device)
        input_ids_base = input_ids.to(md.device)
        attention_mask_base = attention_mask.to(md.device)
        # print('reached  here as well')
        # print('input_ids_base device', input_ids_base.device)
        # print('self.ft_model device', model_ft.device)
        outputs_base = md.forward(input_ids_base, attention_mask=attention_mask_base, position_ids=position_ids_base)
        # outputs_base = self.ft_model.forward(input_ids, attention_mask=attention_mask)
        # print(type(outputs_base))
        logits_base = outputs_base.logits
        # Convert logits to probabilities (apply softmax)
        probabilities_base = torch.nn.functional.softmax(logits_base, dim=-1)
    return probabilities_base # [batch_size, seq_len, vocab_size]