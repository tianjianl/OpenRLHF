#!/bin/bash
#SBATCH --partition a100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name="dpo_pipeline"
#SBATCH --output="slurm_logs/dpo_pipeline_bo4_llama3_full.log"
#SBATCH --mem=200G
#SBATCH --exclude=c001,c003,c005,c013
source ~/.bashrc

cd /home/tli104/OpenRLHF/scripts/

#bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.0
#bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 1.0
#bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.5
#bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.8
#bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.2
bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.7
bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.3
bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.6
bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.4
bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.9
bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized meta-llama/Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.1

#bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized allenai/Llama-3.1-Tulu-3-8B-SFT 12 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.0
#bash pipeline.sh HuggingFaceH4/ultrafeedback_binarized allenai/Llama-3.1-Tulu-3-8B-SFT 2 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 1.0
#bash examples/scripts/train_weighted_sft_custom.sh allenai/Llama-3.1-Tulu-3-8B-SFT dogtooth/ultrafeedback_weighted_sft 5e-7

