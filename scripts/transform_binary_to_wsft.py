import argparse
from datasets import load_dataset, Dataset
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Transform dataset for weighted SFT.")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Hugging Face dataset path (e.g., 'my_org/my_dataset').")
    parser.add_argument("--original_split", type=str, default="train", 
                        help="Split of the dataset to read and transform. Default: 'train'.")
    parser.add_argument("--new_dataset_suffix", type=str, default="_weighted_sft",
                        help="Suffix to add to the original dataset name for the new dataset.")
    parser.add_argument("--new_split", type=str, default="train_weighted_sft", 
                        help="New split name in the new dataset. Default: 'train_weighted_sft'.")
    parser.add_argument("--min_score", type=float, default=-1, 
                        help="Minimum score for normalization. Default: -1.")
    parser.add_argument("--max_score", type=float, default=1,
                        help="Maximum score for normalization. Default: 1.")
    
    args = parser.parse_args()

    # Load the entire dataset (all splits)
    ds = load_dataset(args.dataset)

    # Extract and transform only the specified original split
    original_dataset = ds[args.original_split]

    # Extract scores for normalization
    all_scores = []
    for example in original_dataset:
        chosen_score_val = list(example["chosen_score"].values())[0]
        rejected_score_val = list(example["rejected_score"].values())[0]
        
        all_scores.extend([chosen_score_val, rejected_score_val])

    # Compute min/max for normalization
    min_score = min(all_scores)
    max_score = max(all_scores)

    
    def normalize_score(s, a=-1, b=1):
        # Maps s in [min_score, max_score] to [a, b]
        return (s - min_score) * ((b - a) / (max_score - min_score)) + a

    # Transform the dataset
    new_rows = []
    for example in original_dataset:
        chosen_score_val = list(example["chosen_score"].values())[0]
        rejected_score_val = list(example["rejected_score"].values())[0]
        
        # Extract prompt from chosen[0]["content"]
        prompt = example["chosen"][0]["content"]

        # Extract the response from the last message in chosen and rejected
        chosen_response = example["chosen"][-1]["content"]
        rejected_response = example["rejected"][-1]["content"]

        # Add chosen row
        new_rows.append({
            "prompt": prompt,
            "response": example["chosen"],
            "normalized_score": normalize_score(chosen_score_val, args.min_score, args.max_score),
            "raw_score": chosen_score_val,
            "clipped_score": np.clip(chosen_score_val, args.min_score, args.max_score),
        })

        # Add rejected row
        new_rows.append({
            "prompt": prompt,
            "response": example["rejected"],
            "normalized_score": normalize_score(rejected_score_val, args.min_score, args.max_score),
            "raw_score": rejected_score_val,
            "clipped_score": np.clip(rejected_score_val, args.min_score, args.max_score),
        })

    # Create new dataset from transformed rows
    new_dataset = Dataset.from_list(new_rows)
    
    # Build the new dataset repository name
    # If args.dataset is 'my_org/my_dataset', the new dataset will be 'my_org/my_dataset_weighted_sft'
    new_dataset_path = f"dogtooth/{args.dataset.split('/')[-1]}{args.new_dataset_suffix}"
    
    from datasets import DatasetDict
    new_dataset.push_to_hub(new_dataset_path)

if __name__ == "__main__":
    main()
