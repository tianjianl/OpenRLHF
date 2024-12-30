set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset dogtooth/single_pairwise_sft \
   --input_key prompt \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3.1-8B-Instruct \
   --save_path ${SCRATCH_DIR}/judge_checkpoints/llama31-8b-sft-s+p \
   --save_steps 400 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-7 \
   --load_checkpoint \
   --gradient_checkpointing \
   --apply_chat_template \
   --tokenizer_chat_template meta-llama/Meta-Llama-3.1-8B-Instruct-5e-7 \
   --use_wandb $WANDB_API_KEY
EOF
    # --use_wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
