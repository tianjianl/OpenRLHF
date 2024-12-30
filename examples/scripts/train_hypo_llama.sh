#!/bin/bash
#SBATCH --partition a100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c001
#SBATCH --job-name="dpo"
#SBATCH --output="slurm_logs/tulu_hs_off_policy_then_on_policy.log"
#SBATCH --mem=200G

source ~/.bashrc
conda activate /scratch/dkhasha1/tli104/openrlhf_env
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ${SCRATCH_DIR}/open_rlhf_checkpoints/tulu-8b-dpo-hs-off-policy-5e-7-then-on-policy \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain ${SCRATCH_DIR}/open_rlhf_checkpoints/tulu-8b-dpo-hs-off-policy-5e-7 \
   --bf16 \
   --max_epochs 1 \
   --max_len 1024 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset dogtooth/tulu_8b_generated_gold_scored_uf \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples 
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)
if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ${SCRATCH_DIR}/open_rlhf_checkpoints/tulu-8b-dpo-hs-off-policy-2e-7 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain allenai/Llama-3.1-Tulu-3-8B-SFT \
   --bf16 \
   --max_epochs 1 \
   --max_len 1024 \
   --zero_stage 3 \
   --learning_rate 2e-7 \
   --beta 0.1 \
   --dataset dogtooth/helpsteer2_binarized \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples 
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)
if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ${SCRATCH_DIR}/open_rlhf_checkpoints/tulu-8b-dpo-hs-off-policy-1e-6 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain allenai/Llama-3.1-Tulu-3-8B-SFT \
   --bf16 \
   --max_epochs 1 \
   --max_len 1024 \
   --zero_stage 3 \
   --learning_rate 1e-6 \
   --beta 0.1 \
   --dataset dogtooth/helpsteer2_binarized \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples 
EOF
 

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
