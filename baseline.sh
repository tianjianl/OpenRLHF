#!/bin/bash
#SBATCH --partition a100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c001,c011
#SBATCH --time=72:00:00
#SBATCH --job-name="rej_sam_dpo"
#SBATCH --output="iter_dpo_baseline_hs.log"
#SBATCH --mem=200G
#SBATCH --mail-user=tli104@jhu.edu

source ~/.bashrc
conda activate /scratch/dkhasha1/tli104/openrlhf_env
export PYTHONPATH=/weka/home/tli104/OpenRLHF:$PYTHONPATH
bash examples/scripts/train_iterative_dpo_llama.sh
