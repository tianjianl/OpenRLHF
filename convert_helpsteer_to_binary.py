import json
import uuid
from datasets import load_dataset
from huggingface_hub import create_repo, HfApi

# Function to compute the weighted score based on given weights
def compute_weighted_score(response):
    return (0.65 * response["helpfulness"] +
            0.8 * response["correctness"] +
            0.45 * response["coherence"] +
            0.55 * response["complexity"] -
            0.4 * response["verbosity"])

# Function to compare two responses based on the weighted score and tie-breaking criteria
def choose_higher_score_response(response1, response2):
    score1 = compute_weighted_score(response1)
    score2 = compute_weighted_score(response2)
    
    # Choose the response with the higher weighted score
    if score1 > score2:
        return response1, response2  # response1 is chosen, response2 is rejected
    elif score2 > score1:
        return response2, response1  # response2 is chosen, response1 is rejected
    else:
        return -1, -1

# Function to format the messages as user/assistant conversation
def format_messages(prompt, response):
    return [
        {"content": prompt, "role": "user"},
        {"content": response, "role": "assistant"}
    ]

# Function to process the dataset and convert it into the required format
def process_helpsteer_dataset(helpsteer_dataset, output_file):
    # Group responses by the same prompt
    prompt_responses = {}
    for entry in helpsteer_dataset:
        prompt = entry["prompt"]
        if prompt not in prompt_responses:
            prompt_responses[prompt] = []
        prompt_responses[prompt].append(entry)
    
    # Prepare the list to store the output data
    output_data = []
    
    # Process each prompt and its responses
    for prompt, responses in prompt_responses.items():
        if len(responses) < 2:
            # Skip prompts with less than 2 responses since we can't form a pair
            continue
        
        # Iterate over pairs of responses for each prompt
        for i in range(len(responses) - 1):
            # Choose the response with the higher score as the chosen response
            chosen, rejected = choose_higher_score_response(responses[i], responses[i+1])
            if chosen == -1: 
                break
           
            # Compute the scores for the chosen and rejected responses
            score_chosen = compute_weighted_score(chosen)
            score_rejected = compute_weighted_score(rejected)
            
            # Randomly generate a prompt_id
            prompt_id = str(uuid.uuid4())
            
            # Prepare the formatted output with both chosen and rejected responses wrapped in the message format
            formatted_entry = {
                "prompt": prompt,
                "prompt_id": prompt_id,
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                "messages_chosen": format_messages(prompt, chosen["response"]),
                "messages_rejected": format_messages(prompt, rejected["response"]),
                "score_chosen": score_chosen,
                "score_rejected": score_rejected
            }
            
            # Append to output data
            output_data.append(formatted_entry)
    
    # Save the converted data to a JSON file
    with open(output_file, 'w') as f_out:
        json.dump(output_data, f_out, indent=4)

# Function to upload dataset to Hugging Face
def upload_to_huggingface(output_file, repo_name):
    # Create a dataset repository on Hugging Face
    create_repo(repo_name, repo_type="dataset", exist_ok=True)  
    # Upload the file to the created dataset repository
    api = HfApi()
    api.upload_file(
        path_or_fileobj=output_file,
        path_in_repo=output_file,  # The name for the file in the repo
        repo_id=repo_name,  # Your dataset repository
        repo_type="dataset"
    )
    print(f"Dataset uploaded to Hugging Face at {repo_name}")

# Main function to handle the conversion and upload
def main(repo_name):
    # Load the HelpSteer2 dataset from Hugging Face
    helpsteer_dataset = load_dataset("nvidia/HelpSteer2", split="train")
    
    # Define the output file name
    output_file = "binary_preference_dataset.json"
    
    # Convert the dataset into binary preference format
    process_helpsteer_dataset(helpsteer_dataset, output_file)
    print(f"Dataset converted and saved to {output_file}")
    
    # Upload the dataset to Hugging Face
    upload_to_huggingface(output_file, repo_name)

# Example usage
repo_name = "dogtooth/helpsteer2_binarized"  # Name of the dataset repository on Hugging Face

# Run the main function
main(repo_name)

