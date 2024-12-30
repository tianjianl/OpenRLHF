from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

# Load the original dataset
dataset = load_dataset("dogtooth/ultrafeedback_binarized_scaled")
def transform_split(split_dataset):
    # Create a dictionary for new examples
    transformed_data = {
        "prompt": [],
        "response": [],
        "score": []
    }
    
    # For each example, split it into two rows: one for the chosen and one for the rejected
    for example in split_dataset:
        # Chosen
        transformed_data["prompt"].append(example["prompt"])
        transformed_data["response"].append(example["chosen"])
        transformed_data["score"].append(example["score_chosen"])
        
        # Rejected
        transformed_data["prompt"].append(example["prompt"])
        transformed_data["response"].append(example["rejected"])
        transformed_data["score"].append(example["score_rejected"])
    
    # Convert to a Dataset
    return Dataset.from_dict(transformed_data)

# Apply the transformation to all splits in the dataset
new_dataset_dict = DatasetDict({
    split: transform_split(dataset[split]) for split in ['train_prefs']
})

# (Optional) Print out a few examples to verify the transformation
print(new_dataset_dict["train_prefs"][0])
print(new_dataset_dict["train_prefs"][1])

# Push the dataset to the Hub
# Make sure you're logged in via 'huggingface-cli login'
new_dataset_dict.push_to_hub("dogtooth/ultrafeedback_weighted_sft")

