import pandas as pd
import torch

from datasets import load_dataset, Dataset, DatasetDict, DownloadConfig
from sklearn.model_selection import train_test_split

from utils.data_preprocess import process_e2e_nlg_cleaned, process_web_nlg, process_common_gen, process_nike, process_adidas

class ProcessedDataset():
    def __init__(self, name, base_model_name):
        self.name = name
        self.base_model_name = base_model_name
        print("***")
        print("Start loading dataset")
        print("***")
        
        if(name == 'nike'):
            self.data = process_nike("NikeProductDescriptions.csv", self.base_model_name)
        elif(name == 'adidas'):
            self.data = process_adidas("adidas.csv", self.base_model_name)
        elif(name == 'web_nlg'):
            self.data = load_dataset("web_nlg", 'webnlg_challenge_2017', trust_remote_code=True)
        else:
            self.data = load_dataset(name, trust_remote_code=True)
        
        if(name == 'web_nlg'):
            self.data = process_web_nlg(self.data, self.base_model_name)
        elif(name == 'e2e_nlg_cleaned'):
            self.data = process_e2e_nlg_cleaned(self.data, self.base_model_name)
        elif(name == 'common_gen'):
            self.data = process_common_gen(self.data, self.base_model_name)
            
        print("***")
        print("Finish loading dataset")
        print("***")

    # Preprocess the dataset to include the meaning representation (MR) as input and human reference as target
    def preprocess_sep_input_target(self, examples, tokenizer, 
                                    input_size, target_size, add_extra_tokens = False):
        # Concatenate MR and human reference with a separator
        if(add_extra_tokens):
            inputs = [f"<bos> {mr} <eos>" for mr in examples["meaning_representation"]]
            targets = [f"<bos> {ref} <eos>" for ref in examples["human_reference"]]
        else:
            inputs = [f"{mr}" for mr in examples["meaning_representation"]]
            targets = [f"{ref}" for ref in examples["human_reference"]]
        model_inputs = tokenizer(inputs, max_length=input_size, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=target_size, truncation=True, padding="max_length")
        
        # Replace padding token id's of the labels by -100 so that it's ignored by the loss
        labels["input_ids"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq] 
            for labels_seq in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def preprocess_concat_input_target(self, examples, tokenizer, 
                                       input_size, target_size):
        # Extract the meaning representations (MR) and human references (target text) from the examples
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]
        
        # Tokenize the inputs (meaning representations)
        tokenized_inputs = tokenizer(
            inputs, 
            max_length=input_size, 
            truncation=True, 
            padding="max_length", 
            # return_tensors="pt"  # Use numpy for batch processing
        )
        
        # Tokenize the targets (human references)
        tokenized_targets = tokenizer(
            targets, 
            max_length=target_size, 
            truncation=True, 
            padding="max_length", 
            # return_tensors="pt"  # Use numpy for batch processing
        )
        
        # Concatenate input_ids (MR) and input_ids from the targets (human reference) into one sequence
        # This creates the full sequence: [MR, target] (all tokenized)
        concatenated_input_ids = [
            list(input_seq) + list(target_seq) for input_seq, target_seq in zip(tokenized_inputs["input_ids"], tokenized_targets["input_ids"])
        ]
        
        # Concatenate attention masks for both MR and target
        concatenated_attention_mask = [
            list(input_mask) + list(target_mask) for input_mask, target_mask in zip(tokenized_inputs["attention_mask"], tokenized_targets["attention_mask"])
        ]
        
        # Prepare the labels for loss computation:
        # We need to ignore the loss for the part corresponding to MR and only compute it for the target (human reference).
        
        labels = []
        for input_len, target_seq in zip([input_size] * len(inputs), tokenized_targets["input_ids"]):
            # Ignore loss for MR part by setting it to -100
            labels_seq = [-100] * input_len
            
            # For the target sequence, we keep the tokens, but set padding tokens to -100
            labels_seq += [token if token != tokenizer.pad_token_id else -100 for token in target_seq]
            
            labels.append(labels_seq)
        
        # Return the final dictionary containing input_ids, attention_mask, and labels
        return {
            "input_ids": torch.tensor(concatenated_input_ids),
            "attention_mask": torch.tensor(concatenated_attention_mask),
            "labels": torch.tensor(labels)
        }
    
    def preprocess_preconcat_input_target(self, examples, tokenizer, 
                                          input_size, target_size):
        # Extract the meaning representations (MR) and human references (target text) from the examples
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]

        # Tokenize the inputs (meaning representations)
        tokenized_inputs = tokenizer(
            inputs, add_special_tokens=True, max_length = input_size, truncation=True, 
        )

        # Tokenize the targets (human references)
        tokenized_targets = tokenizer(
            targets, add_special_tokens=False, max_length = target_size-1, truncation=True, 
        )

        # adding eod token
        # Append eod_token_id to each tokenized target
        for tar in tokenized_targets["input_ids"]:
            tar.append(tokenizer.eos_token_id)
        for tar in tokenized_targets["attention_mask"]:
            tar.append(1)

        tokenized_sentences_inp_ids = []
        for inp, tar in zip(tokenized_inputs['input_ids'], tokenized_targets['input_ids']):
            n_pads = (input_size + target_size) - (len(inp) + len(tar))
            tokenized_sentence_inp_ids = [tokenizer.eos_token_id]*n_pads + inp + tar
            tokenized_sentences_inp_ids.append(tokenized_sentence_inp_ids)
        
        tokenized_sentences_att_mask = []
        for inp, tar in zip(tokenized_inputs['attention_mask'], tokenized_targets['attention_mask']):
            n_pads = (input_size + target_size) - (len(inp) + len(tar))
            tokenized_sentence_att_mask = [0]*n_pads + inp + tar
            tokenized_sentences_att_mask.append(tokenized_sentence_att_mask)
        
        tokenized_sentences = {}
        tokenized_sentences['input_ids'] = tokenized_sentences_inp_ids
        tokenized_sentences['attention_mask'] = tokenized_sentences_att_mask
        
        # sentences = []
        # for inp, tar in zip(inputs, targets):
        #     sentences.append(inp + tar)
        
        # # Tokenize the targets (human references)
        # tokenized_targets = tokenizer(
        #     targets
        # )
        
        # # Tokenize the inputs (meaning representations)
        # tokenized_sentences = tokenizer(
        #     sentences, 
        #     max_length=input_size + target_size, 
        #     truncation=True, 
        #     padding="max_length", 
        #     # return_tensors="pt"  # Use numpy for batch processing
        # )
            
        labels = []
        for comb_seq, target_seq in zip(tokenized_sentences['input_ids'], tokenized_targets['input_ids']):
            label_seq = [-100]*len(comb_seq)
            label_seq[-len(target_seq):] = target_seq
            labels.append(label_seq)
            
        
        # Return the final dictionary containing input_ids, attention_mask, and labels
        return {
            "input_ids": torch.tensor(tokenized_sentences['input_ids']),
            "attention_mask": torch.tensor(tokenized_sentences['attention_mask']),
            "labels": torch.tensor(labels)
        }

    
    def preprocess_input_for_evaluate(self, examples, tokenizer, input_size):
        print('there2')
        inputs = examples["meaning_representation"]
        tokenized = tokenizer(inputs, 
                              return_tensors="pt", 
                              max_length=input_size, truncation=True, padding="max_length")
        print('....', tokenized['input_ids'].squeeze(0))
        # Return the tokenized inputs along with any other features (like labels)
        return {
            'input_ids': tokenized['input_ids'],  # Remove batch dimension
            'attention_mask': tokenized['attention_mask'],
            # 'meaning_representation' : examples['meaning_representation'].squeeze(0)# Remove batch dimension
        }
    
    def direct_tokenize(self, examples, tokenizer, text_size):
        return tokenizer(examples["meaning_representation"], text_target=examples["human_reference"], 
                         padding="max_length", truncation=True, max_length=text_size, 
                         add_special_tokens=True)
    
    def mapped_tokenize(self, tokenizer, input_size, target_size):
        fn_kwargs_dict={"tokenizer": tokenizer, "input_size": input_size, 'target_size' : target_size}
        # this is padding, mr, padding, hr
        # self.data = self.data.map(self.preprocess_concat_input_target, 
        #                           batched=True, 
        #                           fn_kwargs=fn_kwargs_dict,
        #                             # remove_columns=["meaning_representation", "human_reference"]
        #                         )
        # this is padding, mr, hr
        self.data = self.data.map(self.preprocess_preconcat_input_target, 
                                  batched=True, 
                                  fn_kwargs=fn_kwargs_dict,
                                    # remove_columns=["meaning_representation", "human_reference"]
                                )
        
    def mapped_tokenize_for_evaluate(self, tokenizer, input_size):
        fn_kwargs_dict={"tokenizer": tokenizer, "input_size": input_size}
        print('there')
        self.data = self.data.map(self.preprocess_input_for_evaluate, 
                                  batched=True, 
                                  fn_kwargs=fn_kwargs_dict,
                                    # remove_columns=["meaning_representation", "human_reference"]
                                )
        # for ke in self.data.keys():
        #     self.data[ke]['input_ids'] = torch.cat([example['input_ids'] for example in self.data[ke]], dim=0)
        #     self.data[ke]['attention_mask'] = torch.cat([example['attention_mask'] for example in self.data[ke]], dim=0)
        # print('****', type(self.data['test'][0]['input_ids']))

    def convert_to_tensors(self, example):
        # Convert input_ids and attention_mask from lists to tensors
        example['input_ids'] = torch.tensor(example['input_ids'])
        example['attention_mask'] = torch.tensor(example['attention_mask'])
        return example
    
    def map_convert_to_tensors(self):
        self.data = self.data.map(self.convert_to_tensors)
    
    def remove_cols_in_df(self, dat, remove_columns):
        # remove_columns=["meaning_representation", "human_reference"]
        dat = dat.remove_columns(remove_columns)
        return dat

    def remove_cols_in_dict(self, remove_columns):
        # remove_columns=["meaning_representation", "human_reference"]
        for ke in self.data.keys():
            self.data[ke] = self.data[ke].remove_columns(remove_columns)

    def filter_rows_in_list(self, col_name, valid_list):
        # Apply the filter to each subset (train, test, validation)
        self.data = DatasetDict({
            split: dataset.filter(lambda example: example[col_name] in valid_list) for split, dataset in self.data.items()
        })
        
    def split_data(self, split_key_name, test_size = 0.2, random_state = 42):
                
        # # Convert the tokenized dataset to a pandas DataFrame for easier manipulation
        df = pd.DataFrame(self.data[split_key_name])

        # Step 2: Extract all unique MRs
        unique_mrs = df['meaning_representation'].unique()

        # Step 3: Perform train-test split on the unique MRs
        train_mrs, hypervalidation_mrs = train_test_split(unique_mrs, test_size=test_size, 
                                                          random_state=random_state)

        # Step 4: Create new DataFrames for train and hypervalidation based on the split MRs
        train_df = df[df['meaning_representation'].isin(train_mrs)]
        hypervalidation_df = df[df['meaning_representation'].isin(hypervalidation_mrs)]

        # Step 5: Convert back to the Dataset format for Hugging Face
        train_dataset = DatasetDict({split_key_name: Dataset.from_pandas(train_df)})
        train_dataset = self.remove_cols_in_df(train_dataset, '__index_level_0__')
        hypervalidation_dataset = DatasetDict({f'hyper{split_key_name}': Dataset.from_pandas(hypervalidation_df)})
        hypervalidation_dataset = self.remove_cols_in_df(hypervalidation_dataset, '__index_level_0__')


        self.data[split_key_name] = train_dataset[split_key_name]
        self.data[f'hyper{split_key_name}'] = hypervalidation_dataset[f'hyper{split_key_name}']

        # tokenized_dataset = {}
        # # Update tokenized_dataset with the new split
        # tokenized_dataset[train_tag] = train_dataset['train'].remove_columns(["meaning_representation", "human_reference", "__index_level_0__"])

        # tokenized_dataset[validation_tag] = hypervalidation_dataset['hypervalidation'].remove_columns(["meaning_representation", "human_reference", "__index_level_0__"])


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
        inputs = f'{example["meaning_representation"]}'
        tokenized = self.tokenizer(inputs, return_tensors="pt", max_length=self.input_size, truncation=True, padding="max_length")
        
        
        # Return the tokenized inputs along with any other features (like labels)
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'meaning_representation' :   example['meaning_representation']# Remove batch dimension
        }
