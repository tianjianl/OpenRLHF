#!/bin/bash
#SBATCH --partition a100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name="dpo"
#SBATCH --output="slurm_logs/tulu_dpo_new_beta-0.5.log"
#SBATCH --mem=200G
#SBATCH --exclude=c001,c003,c005,c013

source ~/.bashrc
conda activate /scratch/dkhasha1/tli104/openrlhf_env
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ${SCRATCH_DIR}/open_rlhf_checkpoints/tulu-8b-dpo-new-uf-beta-0.5 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --pretrain allenai/Llama-3.1-Tulu-3-8B-SFT \
   --bf16 \
   --max_epochs 1 \
   --max_len 2048 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.5 \
   --dataset dogtooth/uf_Llama-3.1-Tulu-3-8B-SFT_2_Skywork-Reward-Gemma-2-27B-v0.2 \
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


