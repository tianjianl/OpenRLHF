from datasets import load_dataset, DatasetDict
import numpy as np

def main():
    # Step 1: Load the dataset
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
    print(f"Loading dataset '{dataset_name}'...")
    try:
        dataset = load_dataset(dataset_name)
        print("Dataset loaded successfully.\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Step 2: Compute global min and max for 'score_chosen' and 'score_rejected'
    print("Computing global min and max for 'score_chosen' and 'score_rejected'...")
    global_min = float('inf')
    global_max = float('-inf')

    for split in dataset.keys():
        scores_chosen = dataset[split]["score_chosen"]
        scores_rejected = dataset[split]["score_rejected"]

        split_min = min(min(scores_chosen), min(scores_rejected))
        split_max = max(max(scores_chosen), max(scores_rejected))

        if split_min < global_min:
            global_min = split_min
        if split_max > global_max:
            global_max = split_max

        print(f"  Split '{split}': min={split_min}, max={split_max}")

    print(f"\nGlobal min: {global_min}")
    print(f"Global max: {global_max}\n")

    # Check to avoid division by zero
    if global_max == global_min:
        raise ValueError("All scores are identical; cannot perform scaling.")

    # Step 3: Define the scaling function
    def rescale_scores(example):
        # Linear transformation: scaled = ((value - min) / (max - min)) * 2 - 1
        example["score_chosen"] = ((example["score_chosen"] - global_min) / (global_max - global_min)) * 2 - 1
        example["score_rejected"] = ((example["score_rejected"] - global_min) / (global_max - global_min)) * 2 - 1
        return example

    # Step 4: Apply the scaling function to all splits
    print("Applying scaling to all splits...")
    rescaled_dataset = dataset.map(rescale_scores, batched=False)
    print("Scaling applied successfully.\n")

    # Optional: Verify the scaling
    print("Verifying the scaling...")
    for split in rescaled_dataset.keys():
        scores_chosen = rescaled_dataset[split]["score_chosen"]
        scores_rejected = rescaled_dataset[split]["score_rejected"]
        min_chosen = min(scores_chosen)
        max_chosen = max(scores_chosen)
        min_rejected = min(scores_rejected)
        max_rejected = max(scores_rejected)
        print(f"  Split '{split}':")
        print(f"    score_chosen - min: {min_chosen:.4f}, max: {max_chosen:.4f}")
        print(f"    score_rejected - min: {min_rejected:.4f}, max: {max_rejected:.4f}")
    print()

    # Step 5: Upload the rescaled dataset to Hugging Face Hub
    new_repo_name = "dogtooth/ultrafeedback_binarized_scaled"
    print(f"Uploading the rescaled dataset to Hugging Face Hub as '{new_repo_name}'...")

    try:
        # Push the dataset to Hugging Face Hub
        rescaled_dataset.push_to_hub(new_repo_name)
        print(f"Dataset successfully pushed to Hugging Face Hub at '{new_repo_name}'.")
    except Exception as e:
        print(f"Error pushing dataset to Hugging Face Hub: {e}")

if __name__ == "__main__":
    main()

