set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset dogtooth/helpsteer2_llm_judge_sft_rationales_llama31_filtered \
   --input_key prompt \
   --output_key model_completion \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3.1-8B-Instruct \
   --save_path ${SCRATCH_DIR}/judge_checkpoints/llama31-8b-sft-with-rationale \
   --save_steps 400 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --apply_chat_template \
   --tokenizer_chat_template meta-llama/Meta-Llama-3.1-8B-Instruct \
   --use_wandb $WANDB_API_KEY
EOF
    # --use_wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
