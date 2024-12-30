import json
import ray
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import vllm
# Initialize Ray
ray.init()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="allenai/Llama-3.1-Tulu-3-8B-SFT")
parser.add_argument("--dataset", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
parser.add_argument("--num_generations", type=int, default=4)
parser.add_argument("--generation_length", type=int, default=2048)
parser.add_argument("--diverse", action="store_true", help="increase generation diversity")
parser.add_argument("--output_file", type=str, default="output.jsonl")
parser.add_argument("--upload_to_hub", type=str, default=None)
parser.add_argument("--debug", action="store_true", help="only using top 16 prompts, for debugging purposes")
args = parser.parse_args()

# Define the remote LLM class
@ray.remote
class LLM:
    def __init__(self, *args, **kwargs):
        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.llm.encode(*args, **kwargs)

# Define the LLMs class to manage multiple LLM instances
class LLMs:
    def __init__(
        self, *args, tensor_parallel_size=1, pipeline_parallel_size=1, **kwargs
    ):
        num_gpu = torch.cuda.device_count()
        num_gpu_per_llm = tensor_parallel_size * pipeline_parallel_size
        self.llms = []

        funcs = {}
        for key, value in kwargs.items():
            if key.startswith("func_of_"):
                assert callable(
                    value
                ), "value of arguments starting with 'func_of_' should be callable"
                funcs[key[len("func_of_") :]] = value

        for key in funcs:
            kwargs.pop(f"func_of_{key}")

        for idx in range(num_gpu // num_gpu_per_llm):
            other_kwargs = {key: value(idx) for key, value in funcs.items()}

            # GPU allocation and scheduling strategy
            num_gpus = int(tensor_parallel_size == 1)
            scheduling_strategy = None

            if tensor_parallel_size > 1:
                pg = ray.util.placement_group(
                    [{"GPU": 1, "CPU": 1} for _ in range(num_gpu_per_llm)]
                )
                scheduling_strategy = (
                    ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_capture_child_tasks=True,
                        placement_group_bundle_index=0,
                    )
                )
            llm = LLM.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                *args,
                **kwargs,
                **other_kwargs,
            )
            self.llms.append(llm)

    def __len__(self):
        return len(self.llms)

    def __getitem__(self, index):
        return self.llms[index]

# Load the dataset
if args.dataset == 'HuggingFaceH4/ultrafeedback_binarized':
    train_split = 'train_prefs'
else:
    train_split = 'train'

if args.debug:
    dataset = load_dataset(args.dataset, split=f'{train_split}[:16]')
else:
    dataset = load_dataset(args.dataset, split=f'{train_split}')

# Define the function to transform the dataset
def append_prompt(example):
    example["generation_prompt"] = (
        f"Please provide two diverse responses to the following prompt. "
        f"The two responses should differ in aspects such as style, tone, verbosity, formality, structure, emphasis,"
        f"but are not limited to these aspects.\n\n"
        f"Prompt: {example['prompt']}\n\n"
        f"Response 1:"
    )
    return example

# Transform the dataset using map
if args.diverse:
    transformed_dataset = dataset.map(append_prompt)
else:
    transformed_dataset = dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Initialize multiple LLM instances with 2 GPUs each
llms = LLMs(
    model=args.model,
    tensor_parallel_size=2,
    trust_remote_code=True,
    func_of_seed=lambda idx: idx,
)

# Split the dataset into parts using indexing
num_splits = len(llms)
dataset_size = len(transformed_dataset)
split_size = dataset_size // num_splits

splits = []
for i in range(num_splits):
    start_index = i * split_size
    if i == num_splits - 1:
        end_index = dataset_size
    else:
        end_index = (i + 1) * split_size
    splits.append(transformed_dataset[start_index:end_index])

# Function to process a dataset split with a given LLM instance
@ray.remote
def process_split(llm, data_split):
    results = []

    # Collect all generation prompts and prompts
    #generation_prompts = [example for example in data_split["generation_prompt"]]
    prompts = [example for example in data_split["prompt"]]
    generation_prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': example}], tokenize=False) for example in data_split["prompt"]]
    sampling_params = vllm.SamplingParams(
        n=args.num_generations,
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.generation_length,
    )

    # Generate the outputs using VLLM
    outputs = ray.get(llm.generate.remote(generation_prompts, sampling_params))

    # Process the outputs
    for prompt, output in zip(prompts, outputs):
        
        # Parse the generated text to extract the two responses
        if args.diverse:
            generated_text = output.outputs[0].text
        
            split_text = generated_text.split('Response 2:')
            response1 = split_text[0].strip() if len(split_text) > 0 else generated_text.strip()
            response2 = split_text[1].strip() if len(split_text) > 1 else None

            # Create entries for each response
            for idx, response in enumerate([response1, response2], start=1):
                if response:
                    result = {
                        'prompt': prompt,
                        'messages': [
                            {'content': prompt, 'role': 'user'},
                            {'content': response, 'role': 'assistant'}
                        ],
                        'model completion': response,
                        'reference_completion': None  # Assuming no reference completion is available
                    }
                    results.append(result)

        else:
            assert len(output.outputs) == args.num_generations, f"Number of generations {len(output.outputs)} should match the number of requested generations {args.num_generations}"
            for completion in output.outputs:
                generated_text = completion.text

                # remove chat template from generated text
                #chat_template = '<|start_header_id|>assistant<|end_header_id|>\n\n'
                generated_text = generated_text.replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '', 1)
            
                result = {
                    "prompt": prompt,
                    "messages": [
                        {"content": prompt, "role": "user"},
                        {"content": generated_text, "role": "assistant"},
                    ],
                    "model_completion": generated_text,
                    "reference_completion": None
                }
                results.append(result)
    return results

# Process each split with its corresponding LLM instance
futures = []
for idx, data_split in enumerate(splits):
    llm = llms[idx]
    futures.append(process_split.remote(llm, data_split))

# Collect the results
all_results = ray.get(futures)

# Flatten the list of results
all_results = [item for sublist in all_results for item in sublist]

# map the prompts to reference completions
reference_completion_dict = {example['prompt']: example['chosen'][-1]["content"] for example in dataset}

results_dict = {
    'prompt': [result['prompt'] for result in all_results],
    'messages': [result['messages'] for result in all_results],
    'model_completion': [result['model_completion'] for result in all_results],
    'reference_completion': [reference_completion_dict[result['prompt']] for result in all_results]
}


# convert results_dict to a list of dicts
all_results = []
for i in range(len(results_dict['prompt'])):
    result = {
        'prompt': results_dict['prompt'][i],
        'messages': results_dict['messages'][i],
        'model_completion': results_dict['model_completion'][i],
        'reference_completion': results_dict['reference_completion'][i]
    }
    all_results.append(result)

# Save the results to a JSON file
with open(args.output_file, "w") as f:
    for result in all_results:
        f.write(json.dumps(result) + "\n")

new_dataset = Dataset.from_dict(results_dict)

# upload to hub
if args.upload_to_hub:
    new_dataset.push_to_hub(args.upload_to_hub)

