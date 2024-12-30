from datasets import load_dataset, DatasetDict, concatenate_datasets

# Load the datasets
dataset1 = load_dataset("dogtooth/helpsteer2_llm_judge_sft_rationales_llama31_filtered")
dataset2 = load_dataset("dogtooth/helpsteer2_preference_sft")

# Extract "prompt", "response", and add "type": "single" to dataset1
def select_fields_dataset1(example):
    return {
        "prompt": example["prompt"],
        "response": example["model_completion"],
        "type": "single"
    }

dataset1 = dataset1.map(select_fields_dataset1, remove_columns=dataset1['train'].column_names)

# Extract "prompt", "response", and add "type": "pairwise" to dataset2
def select_fields_dataset2(example):
    return {
        "prompt": example["prompt"],
        "response": example["response"],
        "type": "pairwise"
    }

dataset2 = dataset2.map(select_fields_dataset2, remove_columns=dataset2['train'].column_names)

# Concatenate the datasets
combined_dataset = DatasetDict()

for split in set(dataset1.keys()).union(dataset2.keys()):
    datasets_to_concat = []
    if split in dataset1:
        datasets_to_concat.append(dataset1[split])
    if split in dataset2:
        datasets_to_concat.append(dataset2[split])
    combined_split = concatenate_datasets(datasets_to_concat)
    # Shuffle the combined split
    combined_split = combined_split.shuffle(seed=42)
    combined_dataset[split] = combined_split

# Save the combined dataset
combined_dataset.push_to_hub("dogtooth/single_pairwise_sft")
