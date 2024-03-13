MODEL_PATH=$1
NUM_GPUS=4 
BATCH_SIZE=60
PROMPT_TEMPLATE="alpaca"

python -m eval_vllm.driver \
    --model_name_or_path $MODEL_PATH \
    --batch_size $BATCH_SIZE \
    --tensor_parallel_size $NUM_GPUS \
    --prompt_template $PROMPT_TEMPLATE