import os
import random
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

# Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
# or ensure it's stored in the HUGGINGFACE_TOKEN environment variable
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN') or 'YOUR_HF_TOKEN'

def generate_prompt(question, answer_a, answer_b):
    prompt = f"""
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should only consider the helpfulness, correctness, coherence, and complexity of the responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]"""
    return prompt

def main():
    # Load the HelpSteer2 dataset
    dataset = load_dataset('nvidia/HelpSteer2')

    criteria = ['helpfulness', 'correctness', 'coherence', 'complexity']
    new_dataset = {}

    for split in dataset.keys():
        data = dataset[split]
        new_data = {'prompt': [], 'messages': [], 'label': []}

        i = 0
        while i < len(data) - 1:
            example_a = data[i]
            example_b = data[i + 1]

            # Ensure that both responses correspond to the same prompt
            if example_a['prompt'] != example_b['prompt']:
                i += 1
                continue

            question = example_a['prompt']
            response_a = example_a['response']
            response_b = example_b['response']

            # Calculate adjusted scores
            score_a = sum(example_a[criterion] for criterion in criteria) - example_a['verbosity']
            score_b = sum(example_b[criterion] for criterion in criteria) - example_b['verbosity']

            # Randomly assign responses to Assistant A or B
            if random.random() < 0.5:
                assigned_a = response_a
                assigned_b = response_b
                adjusted_score_a = score_a
                adjusted_score_b = score_b
            else:
                assigned_a = response_b
                assigned_b = response_a
                adjusted_score_a = score_b
                adjusted_score_b = score_a

            # Determine the correct label
            if adjusted_score_a > adjusted_score_b:
                label = '[[A]]'
            elif adjusted_score_b > adjusted_score_a:
                label = '[[B]]'
            else:
                label = '[[C]]'

            # Generate the prompt
            prompt_text = generate_prompt(question, assigned_a, assigned_b)
            answer_text = label

            message = [
                {"content": prompt_text, "role": "user"},
                {"content": answer_text, "role": "assistant"}
            ]

            new_data['prompt'].append(prompt_text)
            new_data['messages'].append(message)
            new_data['label'].append(answer_text)
            
            # Move to the next pair
            i += 2

        new_dataset[split] = Dataset.from_dict(new_data)

    new_dataset_dict = DatasetDict(new_dataset)

    # Push the new dataset to Hugging Face Hub
    new_dataset_dict.push_to_hub('dogtooth/helpsteer2_llm_judge_sft_pairwise')
    print(f"Dataset uploaded to https://huggingface.co/dogtooth/helpsteer2_llm_judge_sft_pairwise")

if __name__ == '__main__':
    main()

