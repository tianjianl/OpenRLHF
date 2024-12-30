#!/bin/bash
#SBATCH --partition a100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name="dpo_pipeline"
#SBATCH --output="slurm_logs/dpo_pipeline_iter.log"
#SBATCH --mem=200G
#SBATCH --exclude=c001,c003,c005,c013
source ~/.bashrc

cd /home/tli104/OpenRLHF/scripts/

bash pipeline_iter.sh HuggingFaceH4/ultrafeedback_binarized allenai/Llama-3.1-Tulu-3-8B-SFT 3 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.0 false tulu_3_uf_iter1_small_beta
bash pipeline_iter.sh HuggingFaceH4/ultrafeedback_binarized ${SCRATCH_DIR}/scaling_law_ckpts/tulu_3_uf_iter1_small_beta 3 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.0 false tulu_3_uf_iter2_small_beta
bash pipeline_iter.sh HuggingFaceH4/ultrafeedback_binarized ${SCRATCH_DIR}/scaling_law_ckpts/tulu_3_uf_iter2_small_beta 3 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.0 false tulu_3_uf_iter3_small_beta
bash pipeline_iter.sh HuggingFaceH4/ultrafeedback_binarized ${SCRATCH_DIR}/scaling_law_ckpts/tulu_3_uf_iter3_small_beta 3 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.0 false tulu_3_uf_iter4_small_beta

