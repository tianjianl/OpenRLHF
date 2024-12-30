from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login

# Uncomment the following line to log in to your Hugging Face account
# login()

# Load the datasets
online_dataset = load_dataset("dogtooth/tulu_8b_generated_gold_scored_uf", split="train")
offline_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

# Initialize a list to hold the new examples
new_examples = []

# Process the online dataset first and create a mapping from prompts to indices
prompt_to_index = {}
for idx, example in enumerate(online_dataset):
    # Extract the prompt from the first message in 'chosen'
    prompt = example['chosen'][0]['content']
    online_chosen = example['chosen']
    online_rejected = example['rejected']
    
    # Create a new example with offline_chosen and offline_rejected initialized to empty strings
    new_example = {
        'prompt': prompt,
        'online_chosen': online_chosen,
        'online_rejected': online_rejected,
        'offline_chosen': '',
        'offline_rejected': ''
    }
    new_examples.append(new_example)
    prompt_to_index[prompt] = idx

# Now process the offline dataset
for example in offline_dataset:
    prompt = example['chosen'][0]['content']
    chosen_response = example['chosen']
    rejected_response = example['rejected']    
# Check if this prompt exists in the online dataset
    if prompt in prompt_to_index:
        idx = prompt_to_index[prompt]
        new_examples[idx]['offline_chosen'] = chosen_response
        new_examples[idx]['offline_rejected'] = rejected_response
    else:
        continue

# remove examples with empty offline_chosen and offline_rejected
new_examples = [example for example in new_examples if example['offline_chosen'] != '' and example['offline_rejected'] != '']


new_example_dict = {
    'prompt': [example['prompt'] for example in new_examples],
    'online_chosen': [example['online_chosen'] for example in new_examples],
    'online_rejected': [example['online_rejected'] for example in new_examples],
    'offline_chosen': [example['offline_chosen'] for example in new_examples],
    'offline_rejected': [example['offline_rejected'] for example in new_examples]
}

# print length of the new dataset
print(len(new_example_dict['prompt']))

# Create a new Dataset from the new example
blended_dataset = Dataset.from_dict(new_example_dict)

# Push the blended dataset to the Hugging Face Hub
blended_dataset.push_to_hub("dogtooth/tulu_hybrid_uf")

