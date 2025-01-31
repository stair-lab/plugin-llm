import pandas as pd
import re
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset

def parse_mr_to_string(mr):
    # Regular expression to capture key-value pairs
    pattern = r'(\w+)\[([^\]]+)\]'
    
    # Find all key-value pairs in the string
    key_value_pairs = re.findall(pattern, mr)
    
    # Create a formatted string of key-value pairs
    formatted_string = ',\n'.join([f'{key} -- {value}' for key, value in key_value_pairs])
    
    return formatted_string

def process_e2e_nlg_cleaned(dataset, base_model_name):

    def parse_mr_to_string(mr):
        # Regular expression to capture key-value pairs
        pattern = r'(\w+)\[([^\]]+)\]'
        
        # Find all key-value pairs in the string
        key_value_pairs = re.findall(pattern, mr)
        
        # Create a formatted string of key-value pairs
        formatted_string = ',\n'.join([f'{key} -- {value}' for key, value in key_value_pairs])
        
        return formatted_string

    def add_input_prompt(examples, base_model_name):
        inp = examples['meaning_representation']
        if('gpt2-medium' in base_model_name):
            prefix_str = 'Given the following aspects of a restaurant, "'
            suffix_str = '", a natural language sentence describing the restuarant is: '
            # prefix_str = 'Generate a restaurant description from the following attributes:\n'
            # suffix_str = '\n\nDescription: '
            new_input = prefix_str + inp + suffix_str
        elif('gpt2-xl' in base_model_name):
            prefix_str = 'Imagine you are writing a one-sentence description for a restaurant, given the following aspects: "'
            suffix_str = '", a human-readable natural language sentence describing the restuarant is: '
            new_input = prefix_str + parse_mr_to_string(inp) + suffix_str
        elif('Llama-3.1-8B' in base_model_name):
            # prefix_str = 'Question: Given the following attributes of a restaurant:\n'
            # suffix_str = ',\nhow would you describe the restaurant based on the attributes? Do not provide explanation.\nAnswer: '
            prefix_str = 'Please convert the following attributes into a coherent sentence. Do not provide explanation.\n\nAttributes:\n'
            suffix_str = '\n\nSentence:\n'
            new_input = prefix_str + parse_mr_to_string(inp) + suffix_str
        else:
            prefix_str = ''
            suffix_str = ''
            new_input = prefix_str + inp + suffix_str
        
        return {'meaning_representation': new_input}
    
    fn_kwargs_dict={"base_model_name": base_model_name}

    dataset['train'] = dataset['train'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    dataset['validation'] = dataset['validation'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    dataset['test'] = dataset['test'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    return dataset

def process_web_nlg(dataset, base_model_name, train_categories = ['Airport', 'Building', 'University', 'Monument', 'MeanOfTransportation'], 
                    test_categories = ['Artist', 'Politician', 'Athlete', 'ComicsCharacter', 'Astronaut', 'SportsTeam' ]):
    
    #Step 0: raname the keys
    # dataset['validation'] = dataset.pop('dev')

    # Step 1: Filter the 'train', 'dev', and 'test' datasets by category
    def filter_by_category(example, categories):
        return example['category'] in categories

    # Filter the 'train' set to only include the "Airport" category
    train_dataset = dataset['train'].filter(lambda x: filter_by_category(x, train_categories))

    # Filter 'dev' and 'test' sets to only include the "Food" category
    dev_dataset = dataset['dev'].filter(lambda x: filter_by_category(x, test_categories))
    test_dataset = dataset['test'].filter(lambda x: filter_by_category(x, test_categories))

    # Step 3: Select only one reference sentence from 'text' field based on 'comment'
    def select_good_comment(example):
        for i, comment in enumerate(example['lex']['comment']):
            if comment == 'good':
                return {'human_reference': example['lex']['text'][i]}  # Pick the sentence marked 'good'
        return {'human_reference': ''}  # Default to the first sentence if none are marked 'good'

    # Apply the function to each dataset
    train_dataset = train_dataset.map(select_good_comment)
    dev_dataset = dev_dataset.map(select_good_comment)
    test_dataset = test_dataset.map(select_good_comment)

    # Step 4: Join 'mtriple_set' list into a string separated by ';'
    def join_mtriple_set(example):
        triples = example['modified_triple_sets']['mtriple_set'][0]
        prompt = ''
        for i, triple in enumerate(triples, start=1):
            prompt += f"{triple}\n"
        # return {'meaning_representation': '\n'.join(example['modified_triple_sets']['mtriple_set'][0])}
        return {'meaning_representation': prompt}

    # Apply the function to each dataset
    train_dataset = train_dataset.map(join_mtriple_set)
    dev_dataset = dev_dataset.map(join_mtriple_set)
    test_dataset = test_dataset.map(join_mtriple_set)

    combined_dev_test = concatenate_datasets([dev_dataset, test_dataset])
    # combined_dev_test = combined_dev_test.train_test_split(test_size=0.5, seed=42)
    # # Rename the splits for clarity
    # combined_dev_test = DatasetDict({
    #     'validation': combined_dev_test['train'],  # Rename 'train' split as 'validation'
    #     'test': combined_dev_test['test']  # Keep 'test' split as 'test'
    # })
    df = combined_dev_test.to_pandas()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    unique_meaning_reps = df["meaning_representation"].unique()
    split_point = len(unique_meaning_reps) // 2

    group1_meaning_reps = set(unique_meaning_reps[:split_point])
    group2_meaning_reps = set(unique_meaning_reps[split_point:])

    group1_df = df[df["meaning_representation"].isin(group1_meaning_reps)]
    group2_df = df[df["meaning_representation"].isin(group2_meaning_reps)]

    group1_dataset = Dataset.from_pandas(group1_df)
    group2_dataset = Dataset.from_pandas(group2_df)

    combined_dev_test = DatasetDict({
        "validation": group1_dataset,
        "test": group2_dataset
    })

    # Step 5: Retain only 'meaning_representation' and 'human_reference' fields
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in ['meaning_representation', 'human_reference']])
    combined_dev_test['validation'] = combined_dev_test['validation'].remove_columns(
        [col for col in combined_dev_test['validation'].column_names if col not in ['meaning_representation', 'human_reference']])
    combined_dev_test['test'] = combined_dev_test['test'].remove_columns(
        [col for col in combined_dev_test['test'].column_names if col not in ['meaning_representation', 'human_reference']])

    # Step 6 removing empty human reference
    # Define a function to filter out rows where 'human_reference' is empty or None
    def filter_empty_human_reference(example):
        return example['human_reference'] is not None and example['human_reference'].strip() != ''

    def filter_empty_meaning_representation(example):
        return example['meaning_representation'] is not None and example['meaning_representation'].strip() != ''

    train_dataset = train_dataset.filter(filter_empty_human_reference)
    combined_dev_test['validation'] = combined_dev_test['validation'].filter(filter_empty_human_reference)
    combined_dev_test['test'] = combined_dev_test['test'].filter(filter_empty_human_reference)

    train_dataset = train_dataset.filter(filter_empty_meaning_representation)
    combined_dev_test['validation'] = combined_dev_test['validation'].filter(filter_empty_meaning_representation)
    combined_dev_test['test'] = combined_dev_test['test'].filter(filter_empty_meaning_representation)
    
    dataset = DatasetDict({
    'train': train_dataset,
    'validation': combined_dev_test['validation'],
    'test': combined_dev_test['test']
    })

    def add_input_prompt(examples, base_model_name):
        inp = examples['meaning_representation']
        if('gpt2-medium' in base_model_name):
            prefix_str = 'Convert the following facts into a coherent sentence:\n\nFacts:\n'
            suffix_str = '\nSentence: '
        elif('gpt2-xl' in base_model_name):
            prefix_str = 'You are given the following facts.\n\nFacts:\n'
            suffix_str = '\nA short, coherent sentence summarizing the facts is: '
        elif('Llama-3.1-8B' in base_model_name):
            prefix_str = 'Do not provide explanation or follow-up. Just convert the following facts of an entity into a coherent sentence.\n\nFacts:\n'
            suffix_str = '\nSentence: '
        new_input = prefix_str + inp + suffix_str
        return {'meaning_representation': new_input}
    
    fn_kwargs_dict={"base_model_name": base_model_name}

    dataset['train'] = dataset['train'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    dataset['validation'] = dataset['validation'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    dataset['test'] = dataset['test'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)

    return dataset

def process_common_gen(dataset, base_model_name):
    
    # Define a function to filter rows
    def filter_target_and_concepts(example):
        # Check if 'man' exists as a whole word in 'target'
        target_contains_man = bool(re.search(r'\bman\b', example['target']))

        # Ensure 'man' and 'woman' are not in 'concepts'
        concepts_does_not_contain_man_or_woman = all(c not in ['man', 'woman'] for c in example['concepts'])

        # Return True if the conditions are met
        return target_contains_man and concepts_does_not_contain_man_or_woman
    
    dataset['train'] = dataset['train'].filter(filter_target_and_concepts)

    def join_concepts(example):
        return {'meaning_representation': ', '.join(example['concepts'])}

    # Apply the function to each dataset
    dataset['train'] = dataset['train'].map(join_concepts)
    dataset['validation'] = dataset['validation'].map(join_concepts)
    dataset['test'] = dataset['test'].map(join_concepts)
    
    def ret_human_reference(example):
        return {'human_reference': example['target']}
    
    dataset['train'] = dataset['train'].map(ret_human_reference)
    dataset['validation'] = dataset['validation'].map(ret_human_reference)
    dataset['test'] = dataset['test'].map(ret_human_reference)

    # # splitting from validation data
    # combined_dev_test = dataset['validation'].train_test_split(test_size=0.5, seed=42)
    # dataset['validation'] = combined_dev_test['train']
    # dataset['test'] = combined_dev_test['test']

    df = dataset['validation'].to_pandas()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    unique_meaning_reps = df["meaning_representation"].unique()
    split_point = len(unique_meaning_reps) // 2

    group1_meaning_reps = set(unique_meaning_reps[:split_point])
    group2_meaning_reps = set(unique_meaning_reps[split_point:])

    group1_df = df[df["meaning_representation"].isin(group1_meaning_reps)]
    group2_df = df[df["meaning_representation"].isin(group2_meaning_reps)]

    group1_dataset = Dataset.from_pandas(group1_df)
    group2_dataset = Dataset.from_pandas(group2_df)
    
    dataset['validation'] = group1_dataset
    dataset['test'] = group2_dataset
    
    dataset['train'] = dataset['train'].remove_columns(
        [col for col in dataset['train'].column_names if col not in ['meaning_representation', 'human_reference']])
    dataset['validation'] = dataset['validation'].remove_columns(
        [col for col in dataset['validation'].column_names if col not in ['meaning_representation', 'human_reference']])
    dataset['test'] = dataset['test'].remove_columns(
        [col for col in dataset['test'].column_names if col not in ['meaning_representation', 'human_reference']])
    
    def add_input_prompt(examples, base_model_name):
        inp = examples['meaning_representation']
        if('gpt2-medium' in base_model_name):
            prefix_str = 'One coherent sentence that uses all the following concepts: '
            suffix_str = ', is: '
        elif('gpt2-xl' in base_model_name):
            prefix_str = 'One coherent sentence that uses all the following concepts: '
            suffix_str = ', is: '
        elif('Llama-3.1-8B' in base_model_name):
            prefix_str = 'Please write a coherent sentence that uses all the following concepts.\n\nConcepts:\n'
            suffix_str = '\nSentence: '
            # suffix_str = '\nSentence:\n'
        new_input = prefix_str + inp + suffix_str
        return {'meaning_representation': new_input}
    
    fn_kwargs_dict={"base_model_name": base_model_name}

    dataset['train'] = dataset['train'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    dataset['validation'] = dataset['validation'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    dataset['test'] = dataset['test'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    
    return dataset

def process_adidas(dataset, base_model_name):
    def convert_to_pair_format(examples):
        # Combine title and subtitle into a meaning representation format
        # Handle cases where subtitle might be missing
        name = examples['name'] if examples['name'] else "None"
        category = examples['category'] if examples['category'] else "None"
        price = examples['selling_price'] if examples['selling_price'] else "None"
        color = examples['color'] if examples['color'] else "None"
        
        mr = f"name[{name}], category[{category}], price[{price}], color[{color}]"
        
        return {
            'meaning_representation': mr,
            'human_reference': examples['description']
        }
    
    # Convert the CSV data into a Dataset
    df = pd.read_csv(dataset)
    
    # Take last 100 samples for test, rest for validation
    test_df = df.tail(100)
    val_df = df.head(len(df) - 100)
    
    # Convert to datasets
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Convert format
    val_dataset = val_dataset.map(convert_to_pair_format)
    test_dataset = test_dataset.map(convert_to_pair_format)
    
    dataset = DatasetDict({
        'validation': val_dataset,
        'test': test_dataset
    })
    
    def add_input_prompt(examples, base_model_name):
        inp = examples['meaning_representation']
        if('gpt2-medium' in base_model_name):
            prefix_str = 'Given the following attributes of a product, write a description.\n\n'
            suffix_str = '\n\nDescription:'
            new_input = prefix_str + inp + suffix_str
        elif('gpt2-xl' in base_model_name):
            prefix_str = 'Given the following attributes of a product, write a description.\n\n'
            suffix_str = '\n\nDescription:'
            new_input = prefix_str + parse_mr_to_string(inp) + suffix_str
        elif('Llama-3.1-8B' in base_model_name):
            prefix_str = 'Please write a description of this product given the following attributes.\n\n'
            suffix_str = '\n\nDescription:\n'
            new_input = prefix_str + parse_mr_to_string(inp) + suffix_str
        else:
            prefix_str = ''
            suffix_str = ''
            new_input = prefix_str + inp + suffix_str
        
        return {'meaning_representation': new_input}
    
    fn_kwargs_dict={"base_model_name": base_model_name}

    dataset['validation'] = dataset['validation'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    dataset['test'] = dataset['test'].map(add_input_prompt, fn_kwargs=fn_kwargs_dict)
    
    return dataset
