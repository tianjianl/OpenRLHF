from datasets import load_dataset, DatasetDict, Dataset

# Load the dataset
preference_dataset = load_dataset("nvidia/HelpSteer2", data_dir="preference")

# Initialize data splits
data_splits = {'train': {'prompt': [], 'response': []}, 'validation': {'prompt': [], 'response': []}}

# Template for the prompt
template = '''Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if @Response 1 is better, "[[B]]"
if @Response 2 is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of @Response 1]
{answer_a}
[The End of @Response 1]
[The Start of @Response 2]
{answer_b}
[The End of @Response 2]'''

# Process the examples
for example in preference_dataset['train']:
    split = example['split']
    if split == 'val':
        split = 'validation'
    # Extract the fields
    question = example['prompt']
    answer_a = example['response_1']
    answer_b = example['response_2']
    preference_elaboration = example['preference_statement'] + " " + example['preference_elaboration']
    pref_strength = example['preference_strength']
    
    # Create the prompt
    prompt = template.format(question=question, answer_a=answer_a, answer_b=answer_b)
    
    # Determine the judgement
    if pref_strength > 0:
        judgement = '[[B]]'
    elif pref_strength < 0:
        judgement = '[[A]]'
    else:
        judgement = '[[C]]'
    
    # Create the response
    response = preference_elaboration + '\n' + judgement
    if len(preference_elaboration) <= 5:
        continue
    
    # Add the examples to the data splits
    data_splits[split]['prompt'].append(prompt)
    data_splits[split]['response'].append(response)

# Create DatasetDict
dataset_dict = DatasetDict()

# Create the datasets
for split, data in data_splits.items():
    dataset = Dataset.from_dict(data)
    dataset_dict[split] = dataset

# Push the dataset to Hugging Face Hub
dataset_dict.push_to_hub('dogtooth/helpsteer2_preference_sft')

