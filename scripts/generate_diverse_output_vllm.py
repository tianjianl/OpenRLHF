
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Load the dataset
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

# Load the tokenizer and model
model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name)

# Iterate over the dataset and generate two diverse outputs for each prompt
for example in dataset['train_prefs']:
    prompt = example['prompt']

    # Create an instruction that clearly asks for two diverse responses
    instruction = (
        f"Please provide two diverse responses to the following prompt. "
        f"The two responses should differ in aspects such as style, tone, verbosity, perspective, "
        f"but are not limited to these aspects.\n\n"
        f"Prompt: {prompt}\n\n"
        f"Response 1:"
    )

    # Set sampling parameters for generation
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048
    )

    # Generate the outputs using VLLM
    output = llm.generate([instruction], sampling_params)
    generated_text = output[0].outputs[0].text

    # Parse the generated text to extract the two responses
    split_text = generated_text.split('Response 2:')
    if len(split_text) >= 2:
        response1 = split_text[0].strip()
        response2 = split_text[1].strip()
    else:
        response1 = generated_text.strip()
        response2 = None
    
    # Print the two responses
    with open('/scratch/dkhasha1/tli104/uf_diverse_output.txt', 'a+') as f:
        print("Prompt:\n", prompt, file=f)
        print("Response 1:\n", response1, file=f)
        print("Response 2:\n", response2, file=f)
        print("====")


