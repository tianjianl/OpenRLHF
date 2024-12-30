#!/usr/bin/env python3

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_log_probabilities(dataset_name, model_name, subset=None, split="train", debug=False):
    """
    Loads a Hugging Face dataset and a causal language model, computes the log-probabilities
    for each row's (prompt -> model_completion), appends a new column 'log_prob', and:
      - If debug = True, only process the first 10 rows, print out the log-probs, and do NOT push to Hub.
      - If debug = False, process entire dataset and push the updated dataset back to the same name on the Hub.
    """

    # 1. Load the dataset (and subset if specified).
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    # If debug, limit dataset to the first 10 rows
    if debug:
        max_rows = min(10, len(dataset))
        dataset = dataset.select(range(max_rows))
        print(f"[DEBUG MODE] Processing only {max_rows} rows from the dataset...")

    # 2. Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def _compute_log_prob(example):
        """
        For each row, compute the log-prob of the model_completion given the prompt.
        Returns a dictionary {'log_prob': value} for dataset.map().
        """

        prompt_text = example["prompt"]
        completion_text = example["model_completion"]

        # Convert prompt and completion to token IDs
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

        # Concatenate prompt+completion for teacher forcing
        input_ids = torch.tensor([prompt_ids + completion_ids], device=device)

        # Forward pass (no gradient) to get logits
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

        # log_softmax along the vocab dimension for each predicted token
        # Shape: [batch_size, seq_len-1, vocab_size]
        log_probs_all_tokens = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)

        # The target tokens are everything except the first token in input_ids
        target_ids = input_ids[:, 1:]  # shape: [batch_size, seq_len-1]

        # Gather the log-prob for the correct target token at each position
        # shape of target_log_probs: [batch_size, seq_len-1]
        target_log_probs = log_probs_all_tokens.gather(2, target_ids.unsqueeze(2)).squeeze(2)

        # The prompt occupies the first len(prompt_ids) tokens, so the completion tokens
        # correspond to the final len(completion_ids) tokens in that sequence.
        prompt_len = len(prompt_ids)
        completion_len = len(completion_ids)

        # For the completion tokens, we want the indices from
        # (prompt_len - 1) up to (prompt_len - 1 + completion_len) in target_log_probs.
        # We use max(0, prompt_len - 1) in case prompt_len=0 edge cases.
        completion_start_idx = max(0, prompt_len - 1)
        completion_end_idx = completion_start_idx + completion_len
        completion_log_probs = target_log_probs[0, completion_start_idx:completion_end_idx]

        # Sum the log-probs for the entire completion
        total_log_prob = completion_log_probs.sum().item()
        return {"log_prob": total_log_prob}

    # 3. Map the function over the dataset. This creates a new 'log_prob' column.
    updated_dataset = dataset.map(_compute_log_prob, batched=True)

    # Debug mode: Print out the first 10 log-prob values and skip pushing to Hub
    if debug:
        print("\n[DEBUG] The computed 'log_prob' for up to the first 10 entries:")
        for i, example in enumerate(updated_dataset):
            print(f"Row {i}: log_prob = {example['log_prob']}")
        print("\n[DEBUG] Skipping push_to_hub() in debug mode.")
    else:
        # 4. Push the updated dataset to the hub (for the entire split).
        updated_dataset.push_to_hub(dataset_name)
        print(f"Successfully pushed updated dataset '{dataset_name}' with new 'log_prob' column.")

def main():
    parser = argparse.ArgumentParser(description="Compute log-prob of model_completion for a HF dataset and re-upload.")
    parser.add_argument("--dataset_name", required=True,
                        help="Hugging Face dataset name, e.g. dogtooth/hs_Meta-Llama-3.1-8B-Instruct_6")
    parser.add_argument("--model_name", required=True,
                        help="Hugging Face model name, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--subset", default=None, help="Subset/configuration of the dataset if applicable.")
    parser.add_argument("--split", default="train", help="Which split of the dataset to use (default: train)")
    parser.add_argument("--debug", action="store_true",
                        help="If set, only process the first 10 rows and print their log-probs without pushing to Hub.")
    args = parser.parse_args()

    compute_log_probabilities(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        subset=args.subset,
        split=args.split,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
