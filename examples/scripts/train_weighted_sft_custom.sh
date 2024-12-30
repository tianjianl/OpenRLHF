set -x
MODEL=$1
DATASET=$2
LR=$3

if [[ $DATASET == *"ultrafeedback"* ]]; then
    DATASET_SUFFIX='uf'
elif [[ $DATASET == *"helpsteer"* ]]; then
    DATASET_SUFFIX='hs'
else
    DATASET_SUFFIX=$(echo $DATASET | sed 's/.*\///')
fi

if [[ $MODEL == *"Tulu"* ]]; then
    MODEL_SUFFIX='tulu'
elif [[ $MODEL == *"Llama"* ]]; then
    MODEL_SUFFIX='llama'
else
    MODEL_SUFFIX=$(echo $MODEL | sed 's/.*\///')
fi

SAVE_TO=${SCRATCH_DIR}/scaling_law_ckpts/${MODEL_SUFFIX}_8b_${DATASET_SUFFIX}_weighted_sft_${LR}_unnormalized
echo "checkpoint saving to $SAVE_TO"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_weighted_sft \
   --max_len 2048 \
   --dataset ${DATASET} \
   --input_key prompt \
   --output_key response \
   --score_key normalized_score \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain ${MODEL} \
   --save_path ${SAVE_TO} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate ${LR} \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
