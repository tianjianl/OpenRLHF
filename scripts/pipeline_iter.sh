source ~/.bashrc

conda activate /scratch/dkhasha1/tli104/open_instruct_env
set -x

DATASET=$1
POLICY_MODEL=$2
NUM_GENERATIONS=$3
REWARD_MODEL=$4
MIX_OFF_POLICY_PROBS=$5
RESET=$6
SAVE_TO=$7

echo "DATASET: $DATASET"
echo "POLICY_MODEL: $POLICY_MODEL"
echo "NUM_GENERATIONS: $NUM_GENERATIONS"
echo "REWARD_MODEL: $REWARD_MODEL"
echo "MIX_OFF_POLICY_PROBS: $MIX_OFF_POLICY_PROBS"
#bash pipeline.sh dogtooth/helpsteer2_binarized meta-llama/Meta-Llama-3.1-8B-Instruct 4 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 0.0#
# This runs a on-policy dpo run of llama-3.1-8b-instruct on helpsteer2 prompts"
# if ultrafeedback is in dataset string
if [[ $DATASET == *"ultrafeedback"* ]]; then
    DATASET_SUFFIX='uf'
elif [[ $DATASET == *"helpsteer"* ]]; then
    DATASET_SUFFIX='hs'
else
    DATASET_SUFFIX=$(echo $DATASET | sed 's/.*\///')
fi

POLICY_MODEL_SUFFIX=$(echo $POLICY_MODEL | sed 's/.*\///')

REWARD_MODEL_SUFFIX=$(echo $REWARD_MODEL | sed 's/.*\///')

if [[ $RESET == "True" ]]; then
    echo "Resetting scaling_laws directory"
    rm -rf ${SCRATCH_DIR}/scaling_laws/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}*
    rm -rf ${SCRATCH_DIR}/scaling_law_ckpts/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX}
    rm -rf ${SCRATCH_DIR}/scaling_law_eval/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX}
fi
# Run the pipeline
# check if ${SCRATCH_DIR}/scaling_laws/$DATASET_SUFFIX_$POLICY_MODEL_SUFFIX_$NUM_GENERATIONS.jsonl exists, if not then exit
if [ ! -f ${SCRATCH_DIR}/scaling_laws/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}.jsonl ]; then
    echo "Rollout file not found, start generating rollouts"
    ray stop
    python3 ray_vllm_generate.py \
        --dataset $DATASET \
        --model $POLICY_MODEL \
        --num_generations $NUM_GENERATIONS \
        --output_file ${SCRATCH_DIR}/scaling_laws/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}.jsonl \
        --upload_to_hub ${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}
fi

cd /home/tli104/open-instruct
# num of gpu = ( NUM_GENERATIONS + 1 ) mod 8
export PYTHONPATH=/home/tli104/open-instruct:$PYTHONPATH
# check if ${SCRATCH_DIR}/scaling_laws/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX}.jsonl exist
if [ ! -f ${SCRATCH_DIR}/scaling_laws/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX}.jsonl ]; then
    # see if $REWARD_MODEL is not 'lprobs'
    if [[ $REWARD_MODEL != "lprobs" ]]; then    
        NUM_GPU=$(for i in {8..2}; do if [ $(((NUM_GENERATIONS + 1)% i)) -eq 0 ]; then echo $i; break; fi; done)
        echo $NUM_GPU
        python3 open_instruct/rejection_sampling/rejection_sampling.py \
            --input_filename ${SCRATCH_DIR}/scaling_laws/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}.jsonl \
            --model_names_or_paths $REWARD_MODEL \
            --save_filename ${SCRATCH_DIR}/scaling_laws/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX}.jsonl \
            --save_filename_scores ${SCRATCH_DIR}/scaling_laws/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX}_scores.jsonl \
            --max_forward_batch_size 16 \
            --num_completions $NUM_GENERATIONS \
            --num_gpu $NUM_GPU \
            --push_to_hub \
            --add_timestamp false \
            --hf_repo_id ${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX} 
        echo "saved to dogtooth/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX}"
    else
        echo "Reward model is lprobs, using log probs as reward"
        python3 compute_log_prob_distributed.py \
            --model_name $POLICY_MODEL \
            --dataset_name $DATASET 
    fi       
fi

cd /home/tli104/OpenRLHF


# if mix of policy probs is not zero
if [[ $MIX_OFF_POLICY_PROBS == 1.0 ]]; then
    DATASET_DPO=${DATASET}
    DPO_PROBS=1.0
elif [[ $MIX_OFF_POLICY_PROBS != 0.0 ]]; then
    DATASET_DPO=dogtooth/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX},${DATASET}
    RAW_RESULT=$(echo "1 - $MIX_OFF_POLICY_PROBS" | bc -l)
    RESULT=$(printf "%.9g\n" "$RAW_RESULT")
    DPO_PROBS=${RESULT},${MIX_OFF_POLICY_PROBS}
else
    DATASET_DPO=dogtooth/${DATASET_SUFFIX}_${POLICY_MODEL_SUFFIX}_${NUM_GENERATIONS}_${REWARD_MODEL_SUFFIX} 
    DPO_PROBS=1.0
fi

echo "DATASET_DPO: $DATASET_DPO"

set +x
conda activate /scratch/dkhasha1/tli104/openrlhf_env

set -x
read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ${SCRATCH_DIR}/scaling_law_ckpts/${SAVE_TO} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain $POLICY_MODEL \
   --bf16 \
   --max_epochs 1 \
   --max_len 2048 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.001 \
   --dataset $DATASET_DPO \
   --apply_chat_template \
   --dataset_probs $DPO_PROBS \
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
    # check if checkpoint exists
    if [ ! -d ${SCRATCH_DIR}/scaling_law_ckpts/${SAVE_TO} ]; then
        echo "Training DPO"
        deepspeed --module $training_commands
    else
        echo "Checkpoint exists, skipping training"
    fi
fi


mkdir -p ${SCRATCH_DIR}/scaling_law_eval/ 
# evaluation 
cd /home/tli104/open-instruct
set +x
conda activate /scratch/dkhasha1/tli104/open_instruct_env
set -x
export PYTHONPATH=/home/tli104/open-instruct:$PYTHONPATH
mkdir -p ${SCRATCH_DIR}/scaling_law_eval/${SAVE_TO}
bash alpaca_eval.sh ${SCRATCH_DIR}/scaling_law_ckpts/${SAVE_TO} > ${SCRATCH_DIR}/scaling_law_eval/${SAVE_TO}/eval_alpaca.log 
bash scripts/lm_eval_custom_script.sh ${SCRATCH_DIR}/scaling_law_ckpts/${SAVE_TO} > ${SCRATCH_DIR}/scaling_law_eval/${SAVE_TO}/eval_benchmarks.log
