import re
from datasets import load_dataset, DatasetDict

def extract_prompt(message):
    """Extract the prompt from the message."""
    return ''.join([item['content'] for item in message if item['role'] == 'user'])

def filter_example(example, processed_prompts):
    """Filter the example according to the specified rules."""
    prompt = extract_prompt(example.get('messages', []))
    if prompt in processed_prompts:
        # Skip if we've already added a response for this prompt
        return None
    reference_completion = example.get('reference_completion', '').strip()
    # Collect model completions
    model_completions = []
    if 'model_completion' in example and example['model_completion']:
        model_completions.append(example['model_completion'].strip())
    if 'model_completions' in example and example['model_completions']:
        model_completions.extend([mc.strip() for mc in example['model_completions']])
    # Compare each model completion to see if it contains the reference completion
    for completion in model_completions:
        if re.search(re.escape(reference_completion), completion):
            processed_prompts.add(prompt)
            return {'prompt': prompt, 'response': completion}
    # If no completion contains the reference, return None to filter out this example
    return None

def main():
    # Load the original dataset
    dataset = load_dataset('dogtooth/helpsteer2_rej_sampling_pairwise_data_llm_judge')

    # Initialize a dictionary to hold the filtered datasets for each split
    filtered_splits = {}

    # Process each split in the dataset
    for split in dataset.keys():
        # Set to keep track of prompts that have been processed
        processed_prompts = set()
        filtered_examples = []
        for example in dataset[split]:
            result = filter_example(example, processed_prompts)
            if result is not None:
                filtered_examples.append(result)
        # Create a new Dataset object with the filtered examples
        if filtered_examples:
            filtered_splits[split] = filtered_examples

    if filtered_splits:
        # Convert the dictionary of lists to a DatasetDict
        filtered_dataset = DatasetDict({
            split: dataset[split].from_list(filtered_splits[split])
            for split in filtered_splits
        })

        # Save the filtered dataset
        filtered_dataset.push_to_hub('dogtooth/helpsteer2_rej_sampling_pairwise_data_llm_judge_filtered')
        print("Filtered dataset saved to 'dogtooth/helpsteer2_rej_sampling_pairwise_data_llm_judge_filtered' on Hugging Face Hub")
    else:
        print("No matching examples found. No dataset was created.")

if __name__ == '__main__':
    main()

