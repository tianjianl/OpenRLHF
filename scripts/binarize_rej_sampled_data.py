import re
from datasets import load_dataset, Dataset, DatasetDict

def extract_prompt(messages):
    """Extract the prompt from the messages."""
    return ''.join([item['content'] for item in messages if item['role'] == 'user'])

def prepare_messages(prompt, completion):
    """Prepare the messages in the required format."""
    return [
        {"content": prompt, "role": "user"},
        {"content": completion, "role": "assistant"}
    ]

def normalize_whitespace(text):
    """Normalize whitespace in the text."""
    text = text.replace('[ [A] ]', '[[A]]')
    text = text.replace('[ [B] ]', '[[B]]')
    return text
def main():
    # Load the original dataset
    dataset = load_dataset('dogtooth/helpsteer2_rej_sampling_pairwise_data_llm_judge')

    # Initialize a dictionary to group entries by prompt
    prompt_entries = {}

    # Collect and group entries by prompt
    for split in dataset.keys():
        for example in dataset[split]:
            prompt = extract_prompt(example.get('messages', []))
            # Normalize the prompt to handle whitespace inconsistencies
            prompt_normalized = normalize_whitespace(prompt)
            if prompt_normalized not in prompt_entries:
                prompt_entries[prompt_normalized] = {
                    'original_prompt': prompt,  # Keep the original prompt for display
                    'reference_completion': example.get('reference_completion', '').strip(),
                    'model_completions': []
                }
            # Collect model completions
            model_completions = []
            if 'model_completion' in example and example['model_completion']:
                model_completions.append(example['model_completion'].strip())
            if 'model_completions' in example and example['model_completions']:
                model_completions.extend([mc.strip() for mc in example['model_completions']])
            # Add to the list of completions for this prompt
            prompt_entries[prompt_normalized]['model_completions'].extend(model_completions)

    # Initialize a list to store the filtered examples
    filtered_examples = []

    # Process each unique prompt
    for prompt_normalized, entry in prompt_entries.items():
        reference_completion = entry['reference_completion']
        reference_normalized = normalize_whitespace(reference_completion)
        correct_completion = None
        incorrect_completion = None
        # Remove duplicates from model completions
        model_completions = list(set(entry['model_completions']))
        # Check each model completion
        for completion in model_completions:
            completion_normalized = normalize_whitespace(completion)
            if re.search(re.escape(reference_normalized), completion_normalized):
                if not correct_completion:
                    correct_completion = completion_normalized
            else:
                if not incorrect_completion:
                    incorrect_completion = completion_normalized
            # Stop if both are found
            if correct_completion and incorrect_completion:
                break
        # If both correct and incorrect completions are found, add them to the dataset
        if correct_completion and incorrect_completion:
            filtered_examples.append({
                'prompt': entry['original_prompt'],
                'chosen': prepare_messages(entry['original_prompt'], correct_completion),
                'rejected': prepare_messages(entry['original_prompt'], incorrect_completion)
            })

    # Create a Dataset from the filtered examples
    filtered_dataset = Dataset.from_list(filtered_examples)
    dataset_dict = DatasetDict({'train': filtered_dataset})

    # Save the dataset to the Hugging Face Hub
    dataset_dict.push_to_hub('dogtooth/helpsteer2_rej_sampling_pairwise_data_llm_judge_filtered_binarized_full')
    print("Filtered dataset uploaded to 'dogtooth/helpsteer2_rej_sampling_pairwise_data_llm_judge_filtered_binarized_full' on Hugging Face Hub")

if __name__ == '__main__':
    main()

