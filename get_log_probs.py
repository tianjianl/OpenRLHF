import json
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def compute_log_prob(model, tokenizer, prompt, response, device):
    """
    Compute log p(response | prompt) for a causal language model.
    This is done by:
    1. Tokenizing prompt and response.
    2. Running model on [prompt + response] to get logits.
    3. Extracting log-probabilities of response tokens conditioned on all prior tokens (including prompt).
    4. Subtract out the prompt tokens so we consider only the response portion.

    Return the sum of log probabilities (in natural log) of the response tokens.
    """
    # Encode prompt and response
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    input_ids = prompt_ids + response_ids

    # Convert to tensor
    input_ids_t = torch.tensor([input_ids], device=device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids_t)
        logits = outputs.logits

    # logits shape: [batch, seq_len, vocab_size]
    # We want the probabilities of the response tokens.
    # The probability of a token at position i is based on logits[i-1].
    # We'll extract logits for the response part:
    # response starts after len(prompt_ids) tokens
    prompt_length = len(prompt_ids)
    response_length = len(response_ids)

    # Get logits for the positions corresponding to response tokens
    # For token at position i in input_ids (0-based), its probability is predicted by logits at [i, ...].
    # The first response token is at index prompt_length. Its probability is predicted by logits[prompt_length].
    # We sum over all response tokens:
    response_logits = logits[0, prompt_length-1:-1, :] if prompt_length > 0 else logits[0, :len(input_ids)-1, :]
    # Actually, let's be careful: 
    # The token at index i is predicted by logits[i-1], so for the response token at prompt_length, 
    # we look at logits[prompt_length-1]. Similarly for the last response token at input_ids[-1], 
    # predicted by logits[len(input_ids)-2].
    # So the indexing for response tokens' logits is [prompt_length-1 : prompt_length-1+response_length]
    # because we have exactly `response_length` tokens to score.
    response_logits = logits[0, prompt_length-1 : prompt_length-1+response_length, :]

    # Now, gather the logits of the actual response tokens
    # response_ids are the tokens we want the probability of
    response_ids_tensor = torch.tensor(response_ids, device=device)
    # Compute log probabilities
    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=response_ids_tensor.unsqueeze(-1)).squeeze(-1)

    # Sum up the log probabilities for all response tokens
    total_log_prob = token_log_probs.sum().item()
    return total_log_prob

def main():
    # Paths and model
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Replace with your model
    dataset_path = "dogtooth/helpsteer2_binarized"  # This can be a local dataset or a huggingface dataset identifier
    output_file = "llama_31_instruct_hs2_lprobs.jsonl"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Load dataset
    # The dataset should have columns "chosen" and "rejected"
    # Each of these is a list of essages, where the prompt is at index 0 and the final answer at index -1
    dataset = load_dataset(dataset_path)["train"]

    # Process dataset and write out results
    with open(output_file, "w", encoding="utf-8") as f:
        for index, example in enumerate(dataset):
            
            if index % 100 == 0:
                print(index)
            
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]

            # Extract prompt and responses
            prompt_chosen = chosen_messages[0]["content"]
            chosen_response = chosen_messages[-1]["content"]
            # Assuming the prompt is the same for chosen and rejected or is also in the rejected list:
            # If rejected also has the same initial prompt, we use that. Otherwise, we assume it's the same prompt.
            prompt_rejected = rejected_messages[0]["content"]
            rejected_response = rejected_messages[-1]["content"]

            # If the prompt should be the same for chosen and rejected scenario:
            # It's often the case in RLHF datasets that chosen and rejected share the same user prompt.
            # If that's the assumption, we just use prompt_chosen:
            prompt = prompt_chosen

            chosen_log_prob = compute_log_prob(model, tokenizer, prompt, chosen_response, device)
            rejected_log_prob = compute_log_prob(model, tokenizer, prompt, rejected_response, device)

            # Write out to file
            out_entry = {
                "chosen": example["chosen"],
                "rejected": example["rejected"],
                "chosen_log_probs": chosen_log_prob,
                "rejected_log_probs": rejected_log_prob
            }
            f.write(json.dumps(out_entry) + "\n")

if __name__ == "__main__":
    main()

