set -x
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m openrlhf.cli.batch_inference_iter \
   --eval_task iterative_generate \
   --pretrain meta-llama/Meta-Llama-3.1-8B-Instruct \
   --max_new_tokens 1024 \
   --prompt_max_len 8192 \
   --dataset nvidia/HelpSteer2 \
   --input_key prompt \
   --apply_chat_template \
   --temperature 1.0 \
   --tp_size 8 \
   --best_of_n 1 \
   --enable_prefix_caching \
   --max_num_seqs 64 \
   --iter 0 \
   --num_iterations 3 \
   --rollout_batch_size 10240 \
   --output_path ${SCRATCH_DIR}/temp_generate.jsonl
