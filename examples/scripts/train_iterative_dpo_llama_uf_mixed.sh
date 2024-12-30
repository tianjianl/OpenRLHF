set -x
export VLLM_WORKER_MULTIPROC_METHOD=spawn

checkSuccess() {
   if [[ $? != 0 ]]; then
      echo "FAILED $1"
      exit 1
   fi
}

mkdir -p ${SCRATCH_DIR}/llama-3-8b-iter-dpo-uf-mixed-ckpt
GENERATE_OUTPUT=${SCRATCH_DIR}/llama-3-8b-iter-dpo-uf-mixed-ckpt/generate
RM_OUTPUT=${SCRATCH_DIR}/llama-3-8b-iter-dpo-uf-mixed-ckpt/rm
MODEL_OUTPUT_PATH=${SCRATCH_DIR}/llama-3-8b-iter-dpo-uf-mixed-ckpt/checkpoints
ITER_LOG_PATH=null

mkdir -p ${GENERATE_OUTPUT}
mkdir -p ${RM_OUTPUT}
mkdir -p ${MODEL_OUTPUT_PATH}

TRAINING_ITERS=5
ROLLOUT_BATCH_SIZE=10240

POLICY_MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
REF_MODEL_PATH=$POLICY_MODEL_PATH

iter=3
if [ -f $ITER_LOG_PATH ]; then
   iter=$(cat $ITER_LOG_PATH)
fi

while (($iter < $TRAINING_ITERS)); do
   echo "Iter: $iter"
   # Use latest model if past first iteration
   if ((iter > 0)); then
      POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH/$(($iter - 1))
   fi

   read -r -d '' generate_commands <<EOF
openrlhf.cli.batch_inference
   --eval_task generate_vllm \
   --pretrain $POLICY_MODEL_PATH \
   --max_new_tokens 2048 \
   --prompt_max_len 2048 \
   --dataset HuggingFaceH4/ultrafeedback_binarized \
   --dataset_split train_prefs \
   --input_key prompt \
   --apply_chat_template \
   --temperature 1.0 \
   --tp_size 8 \
   --best_of_n 8 \
   --enable_prefix_caching \
   --max_num_seqs 64 \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT/${iter}.jsonl
EOF
   echo $generate_commands
   python -m $generate_commands
   checkSuccess "GENERATE"

    read -r -d '' get_rewards_commands <<EOF
openrlhf.cli.batch_inference
   --eval_task rm \
   --pretrain Skywork/Skywork-Reward-Gemma-2-27B-v0.2 \
   --bf16 \
   --max_len 4096 \
   --dataset $GENERATE_OUTPUT/${iter}.jsonl  \
   --dataset_probs 1.0 \
   --zero_stage 0 \
   --post_processor iter_dpo \
   --micro_batch_size 2 \
   --output_path $RM_OUTPUT/${iter}.jsonl
EOF
   echo $get_rewards_commands
   deepspeed --module $get_rewards_commands
   checkSuccess "RM"

   read -r -d '' dpo_commands <<EOF
openrlhf.cli.train_dpo \
   --max_len 4096 \
   --dataset $RM_OUTPUT/${iter}.jsonl,HuggingFaceH4/ultrafeedback_binarized \
   --dataset_probs 0.5,0.5 \
   --keep_high_quality_ratio 0.1 \
   --prompt_key prompt \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain $POLICY_MODEL_PATH \
   --ref_pretrain $REF_MODEL_PATH \
   --save_path $MODEL_OUTPUT_PATH/$iter \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --learning_rate 5e-7 \
   --gradient_checkpointing
EOF
   echo $dpo_commands
   deepspeed --module $dpo_commands
   checkSuccess "DPO"

   iter=$((iter + 1))
   if [[ "$ITER_LOG_PATH" != "null" ]]; then
      echo $iter >$ITER_LOG_PATH
   fi
done
