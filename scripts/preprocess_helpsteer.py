import os
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

# Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
# or ensure it's stored in the HUGGINGFACE_TOKEN environment variable

def generate_prompt(question, answer, criterion):
    
    prompt_mt_bench = f"""
    Please act as an impartial judge and evaluate the {criterion} of the response provided by an AI assistant to the user question displayed below. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 0 to 4 by strictly following this format: "Rating: [[rating]]", for example: "Rating: [[2]]".

    [Question]
    {question}
    [The Start of Assistant’s Answer]
    {answer}
    [The End of Assistant’s Answer]"""

    prompt_solely = f"""
    Please act as an impartial judge and evaluate the response provided by an AI assistant to the user question displayed below, focusing **solely on its {criterion} and nothing else**. Do not consider any other factors. Begin your evaluation by providing a short explanation based only on {criterion}. Be as objective as possible. After providing your explanation, please rate the response on a scale of 0 to 4 strictly based on {criterion} by strictly following this format: "Rating: [[rating]]", for example: "Rating: [[2]]".

    [Question]
    {question}
    [The Start of Assistant’s Answer]
    {answer}
    [The End of Assistant’s Answer]"""

    prompt_solely_no_explanation = f"""
    Please act as an impartial judge and evaluate the response provided by an AI assistant to the user question displayed below, focusing **solely on its {criterion} and nothing else**. Do not consider any other factors. After reading the response, please rate it on a scale of 0 to 4 strictly based on {criterion} by strictly following this format: "Rating: [[rating]]", for example: "Rating: [[2]]".

    [Question]
    {question}
    [The Start of Assistant’s Answer]
    {answer}
    [The End of Assistant’s Answer]"""

    return prompt_solely


def main():
    
    # Load the HelpSteer2 dataset
    dataset = load_dataset('nvidia/HelpSteer2')

    criteria = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    new_dataset = {}

    for split in ['train', 'validation']:
        new_data = {'prompt': [], 'messages': [], 'label': []}
        for example in dataset[split]:
            question = example['prompt']
            answer = example['response']
            for criterion in criteria:
                prompt_text = generate_prompt(question, answer, criterion)
                rating = example[criterion]
                answer_text = f"Rating: [[{rating}]]"
                new_data['prompt'].append(prompt_text)
                message = [
                    {"content": prompt_text, "role": "user"},
                    {"content": answer_text, "role": "assistant"}
                ]

                new_data['messages'].append(message)
                new_data['label'].append(answer_text)



        new_dataset[split] = Dataset.from_dict(new_data)

    new_dataset_dict = DatasetDict(new_dataset)

    # Push the new dataset to Hugging Face Hub
    new_dataset_dict.push_to_hub('dogtooth/helpsteer2_llm_judge_sft_with_rationale')

    print("https://huggingface.co/datasets/dogtooth/helpsteer2_llm_judge_sft_with_rationale")


if __name__ == '__main__':
    main()
