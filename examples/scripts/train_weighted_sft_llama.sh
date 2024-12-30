set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_weighted_sft \
   --max_len 2048 \
   --dataset HuggingFaceH4/ultrafeedback_binarized \
   --input_key prompt \
   --output_key chosen \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain allenai/Llama-3.1-Tulu-3-8B-SFT \
   --save_path ${SCRATCH_DIR}/scaling_law_ckpts/tulu_8b_uf_weighted_sft_5e-6 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
