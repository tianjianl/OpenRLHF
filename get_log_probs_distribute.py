import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
OUTPUT_PATH = "output.jsonl"
BATCH_SIZE = 8
MAX_LENGTH = 2048

def compute_log_probs_for_response(model, prompt_ids, response_ids, device, max_length=2048):
    """
    Compute the log-probability of the response given the prompt using pre-tokenized IDs.
    """
    # Concatenate prompt and response
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        # If too long, truncate from the left
        input_ids = input_ids[-max_length:]
    
    input_ids_tensor = torch.tensor([input_ids], device=device)
    
    with torch.no_grad():
        outputs = model(input_ids_tensor, use_cache=False)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]
        
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    prompt_len = len(prompt_ids)
    response_len = len(response_ids)
    
    # Extract the log-probs for the response tokens
    predicted_log_probs = log_probs[0, prompt_len:(prompt_len+response_len), :]
    response_token_ids = torch.tensor(response_ids, device=device).unsqueeze(0)  # [1, response_len]
    token_log_probs = predicted_log_probs.gather(1, response_token_ids)
    sum_log_probs = token_log_probs.sum().item()
    return sum_log_probs

def collate_fn(examples, tokenizer, max_length=2048):
    chosen_prompt_texts = [ex["chosen"][0]["content"] for ex in examples]
    chosen_response_texts = [ex["chosen"][-1]["content"] for ex in examples]
    rejected_prompt_texts = [ex["rejected"][0]["content"] for ex in examples]
    rejected_response_texts = [ex["rejected"][-1]["content"] for ex in examples]
    
    chosen_prompt_enc = tokenizer(chosen_prompt_texts, add_special_tokens=False, truncation=True, max_length=max_length)
    chosen_response_enc = tokenizer(chosen_response_texts, add_special_tokens=False, truncation=True, max_length=max_length)
    rejected_prompt_enc = tokenizer(rejected_prompt_texts, add_special_tokens=False, truncation=True, max_length=max_length)
    rejected_response_enc = tokenizer(rejected_response_texts, add_special_tokens=False, truncation=True, max_length=max_length)
    
    batch = {
        "chosen_prompt_ids": chosen_prompt_enc["input_ids"],
        "chosen_response_ids": chosen_response_enc["input_ids"],
        "rejected_prompt_ids": rejected_prompt_enc["input_ids"],
        "rejected_response_ids": rejected_response_enc["input_ids"],
        "original_examples": examples
    }
    
    return batch

def main():
    accelerator = Accelerator()
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="train_prefs")
    data_list = list(dataset)
    
    # Prepare model
    model = accelerator.prepare(model)
    
    # Create DataLoader with collate_fn
    # We'll just pass indices to DataLoader and retrieve examples in collate_fn
    indices = list(range(len(data_list)))
    def wrapped_collate_fn(idx_list):
        # Retrieve the actual examples
        examples = [data_list[i] for i in idx_list]
        return collate_fn(examples, tokenizer, max_length=MAX_LENGTH)
    
    dl = DataLoader(indices, batch_size=BATCH_SIZE, shuffle=False, collate_fn=wrapped_collate_fn)

    output_data = []
    device = accelerator.device
    
    for batch in dl:
        chosen_prompt_ids_batch = batch["chosen_prompt_ids"]
        chosen_response_ids_batch = batch["chosen_response_ids"]
        rejected_prompt_ids_batch = batch["rejected_prompt_ids"]
        rejected_response_ids_batch = batch["rejected_response_ids"]
        original_examples = batch["original_examples"]
        
        for i, example in enumerate(original_examples):
            chosen_prompt_ids = chosen_prompt_ids_batch[i]
            chosen_response_ids = chosen_response_ids_batch[i]
            rejected_prompt_ids = rejected_prompt_ids_batch[i]
            rejected_response_ids = rejected_response_ids_batch[i]
            
            chosen_log_prob = compute_log_probs_for_response(
                model, chosen_prompt_ids, chosen_response_ids, device=device, max_length=MAX_LENGTH
            )
            rejected_log_prob = compute_log_probs_for_response(
                model, rejected_prompt_ids, rejected_response_ids, device=device, max_length=MAX_LENGTH
            )
            
            new_example = dict(example)
            new_example["chosen_log_probs"] = chosen_log_prob
            new_example["rejected_log_probs"] = rejected_log_prob
            output_data.append(new_example)
    
    # Only main process writes results
    if accelerator.is_main_process:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

