# train_online_dpo.py
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer, PairRMJudge
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
reward_model = AutoModelForSequenceClassification.from_pretrained("Skywork/Skywork-Reward-Gemma-2-27B-v0.2", num_labels=1)
reward_tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Gemma-2-27B-v0.2")
train_dataset = load_dataset("HuggingfaceH4/ultrafeedback_binarized", split="train_prefs")
training_args = OnlineDPOConfig(output_dir="/scratch/dkhasha1/tli104/online_dpo/", logging_steps=10)
trainer = OnlineDPOTrainer(
    model=model, reward_model=reward_model, reward_processing_class=reward_tokenizer, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()
