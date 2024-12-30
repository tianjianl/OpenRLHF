from datasets import load_dataset, DatasetDict
import re
# Load the dataset
dataset = load_dataset("dogtooth/helpsteer2_preference_sft")

# Define the replacement function with correct order
def modify_text(text):
    # Specific replacements first
    text = text.replace("The Start of @Response 1", "The Start of Assistant A’s Answer")
    text = text.replace("The End of @Response 1", "The End of Assistant A’s Answer")
    text = text.replace("The Start of @Response 2", "The Start of Assistant B’s Answer")
    text = text.replace("The End of @Response 2", "The End of Assistant B’s Answer")
    # General replacements after
    text = text.replace("@Response 1", "Assistant A")
    text = text.replace("@Response 2", "Assistant B")
    return text

# Apply the function to the dataset
def modify_examples(example):
    example["prompt"] = modify_text(example["prompt"])
    example["response"] = modify_text(example["response"])
    
    return example

# Modify the dataset
modified_dataset = dataset.map(modify_examples)

# Push modified dataset to the Hugging Face Hub
modified_dataset.push_to_hub("dogtooth/helpsteer2_preference_sft_modified")

