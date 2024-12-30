#!/usr/bin/env python
# coding: utf-8

"""
Script to:
1. Load a Hugging Face dataset (e.g. "dogtooth/hs_Meta-Llama-3.1-8B-Instruct_6") that has columns
   ["prompt", "messages", "model_completion", "reference_completion"].
2. Use tokenizer.apply_chat_template(...) to wrap prompt/messages into chat-style format.
3. Load a Hugging Face model (e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct").
4. Compute log probabilities of `model_completion` and `reference_completion` **conditioned on the prompt**,
   ignoring the log-probs of the prompt tokens themselves.
5. Store the sum of log-probs for each completion as new columns "log_prob" and "log_prob_ref".
6. Re-upload the updated dataset to Hugging Face with the **same dataset name**.
7. Use a DataLoader approach (no dataset.map) for log-prob calculation, with DataParallel if multiple GPUs.
8. Include a `--debug` flag that, if set, only processes the first 10 rows.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

debug_input_ids = None
def parse_args():
    parser = argparse.ArgumentParser(description="Compute log probabilities for completions given prompts.")
    parser.add_argument("--dataset_name", type=str, default="dogtooth/hs_Meta-Llama-3.1-8B-Instruct_6",
                        help="Name of the Hugging Face dataset to load and re-upload.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Name of the Hugging Face model to use for computing log probabilities.")
    parser.add_argument("--debug", action="store_true",
                        help="If set, only process the first 10 rows for debugging.")
    return parser.parse_args()

class HFCompletionDataset(Dataset):
    """
    A PyTorch Dataset that holds the HF dataset rows, 
    and will be used for batch computation of log_probs for completions.
    """
    def __init__(self, hf_dataset, tokenizer, max_length=2048, ref_mode=False):
        """
        Args:
            hf_dataset: a split of Hugging Face dataset containing
                        "chat_input", "model_completion" or "reference_completion".
            tokenizer: a transformers tokenizer.
            max_length: Max sequence length for tokenization.
            ref_mode: If True, compute log probs for 'reference_completion'; 
                      else for 'model_completion'.
        """
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ref_mode = ref_mode

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        completion_key = "reference_completion" if self.ref_mode else "model_completion"
        completion = row.get(completion_key, "")

        chat_input = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": ""},
            ],
            tokenize=False
        )

        #print(f"chat_input = {chat_input}")

        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": completion},
            ],
            tokenize=False
        )

        #print(f"full_text = {full_text}")
        
        return {
            "chat_input": chat_input,
            "completion": completion,
            "full_text": full_text,
            "index": idx
        }

def collate_fn(batch, tokenizer):
    """
    Collate function to tokenize the combined chat_input + completion strings in batch.
    We'll produce input_ids, attention_mask, and also store the prompt tokens vs. completion tokens boundaries.
    """
    full_texts = [item["full_text"] for item in batch]
    chat_inputs = [item["chat_input"] for item in batch]

    # Tokenize chat_input separately to get the prompt length
    tokenized_prompt = tokenizer(chat_inputs, 
                                 padding=False,
                                 add_special_tokens=False)
    prompt_lengths = [len(ids) for ids in tokenized_prompt["input_ids"]]

    # Tokenize the entire sequence (chat_input + completion)
    tokenized_full = tokenizer(full_texts, 
                               padding=True,
                               max_length=2048,
                               truncation=True,
                               return_tensors="pt", 
                               add_special_tokens=False)

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    indices = [item["index"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": prompt_lengths,
        "indices": indices,
    }

def compute_log_probs(model, dataloader, tokenizer, hf_dataset, output_column_name, debug=False):
    """
    Compute the log probabilities of the completion tokens for each example in hf_dataset,
    ignoring the log-probs of the prompt tokens. We'll add a new column to hf_dataset named
    output_column_name with the sum of log-probs.
   【 """
    
    log_prob_results = [None] * len(hf_dataset)

    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index % 10 == 0:
                print(f"not at batch {index}")
            input_ids = batch["input_ids"].cuda()
            if index == 0 and debug:
                debug_input_ids = input_ids.cpu()
            elif index == 1 and debug:
                print(f"debug_input_ids = {debug_input_ids}")
                print(input_ids.cpu())
                print(tokenizer.decode(debug_input_ids[0]))
                print("=====================")
                print(tokenizer.decode(input_ids.cpu()[0]))
                assert torch.equal(debug_input_ids, input_ids.cpu()), "Input IDs not equal!"
        
            attention_mask = batch["attention_mask"].cuda()
            prompt_lengths = batch["prompt_lengths"]
            indices = batch["indices"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            #if debug:
            #    print(logits)
            
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            for i, idx in enumerate(indices):
                prompt_length = prompt_lengths[i]
                seq_len = attention_mask[i].sum().item()
                completion_logprob = 0.0

                # Summation of log-probs for tokens in the completion region only
                # We rely on the standard next-token approach:
                # log_prob of token t is at log_probs[i, t-1, input_ids[i,t]] if t > 0
                for t in range(prompt_length, seq_len):
                    if t == 0:
                        continue
                    prev_t = t - 1
                    token_id = input_ids[i, t]
                    completion_logprob += log_probs[i, prev_t, token_id].item()

                log_prob_results[idx] = completion_logprob
            
    # check if output_column_name already exists in the dataset
    if output_column_name in hf_dataset.column_names:
        hf_dataset = hf_dataset.remove_columns(output_column_name)
    hf_dataset = hf_dataset.add_column(output_column_name, log_prob_results)
    return hf_dataset

def main():
    args = parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    debug = args.debug

    # Load dataset
    dataset_dict = load_dataset(dataset_name)
    if isinstance(dataset_dict, DatasetDict):
        if "train" in dataset_dict:
            hf_dataset = dataset_dict["train"]
        else:
            split_name = list(dataset_dict.keys())[0]
            hf_dataset = dataset_dict[split_name]
    else:
        hf_dataset = dataset_dict  # Single Dataset object

    if debug:
        hf_dataset = hf_dataset.select(range(min(2, len(hf_dataset))))

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ---- Log-probs for model_completion ----
    model_comp_dataset = HFCompletionDataset(hf_dataset, tokenizer, ref_mode=False)
    model_comp_dataloader = DataLoader(
        model_comp_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )
    hf_dataset = compute_log_probs(model, model_comp_dataloader, tokenizer, hf_dataset, output_column_name="log_prob")

    # ---- Log-probs for reference_completion ----
    ref_comp_dataset = HFCompletionDataset(hf_dataset, tokenizer, ref_mode=True)
    ref_comp_dataloader = DataLoader(
        ref_comp_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )
    hf_dataset = compute_log_probs(model, ref_comp_dataloader, tokenizer, hf_dataset, output_column_name="log_prob_ref")

    # Re-inject updated 'train' split into a DatasetDict for push_to_hub
    new_dataset_dict = DatasetDict({"train": hf_dataset})
    if args.debug:
        print("Debug mode: Not pushing to Hub.")
        for col in hf_dataset.column_names:
            if col != "log_prob_ref":
                continue
            print(f"{col}: {hf_dataset[col][0]}")
            print(f"{col}: {hf_dataset[col][1]}")
    else:
        new_dataset_dict.push_to_hub(dataset_name, token=os.environ.get("HF_API_TOKEN", None))
        print(f"Updated dataset with log probabilities pushed to the hub: https://huggingface.co/datasets/{dataset_name}")

if __name__ == "__main__":
    main()
