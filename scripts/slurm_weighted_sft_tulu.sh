#!/bin/bash

#SBATCH --partition a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name="weighted_sft"
#SBATCH --output="slurm_logs/weighted_sft_tulu.log"
#SBATCH --mem=200G
#SBATCH --exclude=c001,c003,c005,c013

source ~/.bashrc

conda activate /scratch/dkhasha1/tli104/openrlhf_env
cd /home/tli104/OpenRLHF 

bash examples/scripts/train_weighted_sft_custom.sh allenai/Llama-3.1-Tulu-3-8B-SFT dogtooth/ultrafeedback_weighted_sft 5e-7
bash examples/scripts/train_weighted_sft_custom.sh allenai/Llama-3.1-Tulu-3-8B-SFT dogtooth/uf_Llama-3.1-8B-Instruct_2_Skywork-Reward-Gemma-2-27B-v0.2_weighted_sft 5e-7


bash examples/scripts/train_weighted_sft_custom_normalized.sh allenai/Llama-3.1-Tulu-3-8B-SFT dogtooth/ultrafeedback_weighted_sft 5e-7
bash examples/scripts/train_weighted_sft_custom_normalized.sh allenai/Llama-3.1-Tulu-3-8B-SFT dogtooth/uf_Llama-3.1-8B-Instruct_2_Skywork-Reward-Gemma-2-27B-v0.2_weighted_sft 5e-7

bash examples/scripts/train_weighted_sft_custom_clipped.sh allenai/Llama-3.1-Tulu-3-8B-SFT dogtooth/ultrafeedback_weighted_sft 5e-7
bash examples/scripts/train_weighted_sft_custom_clipped.sh allenai/Llama-3.1-Tulu-3-8B-SFT dogtooth/uf_Llama-3.1-8B-Instruct_2_Skywork-Reward-Gemma-2-27B-v0.2_weighted_sft 5e-7
