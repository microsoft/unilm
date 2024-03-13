MODEL_PATH=$1
SAVE_DIR=$2
NUM_GPUS=4 
BATCH_SIZE=60
PROMPT_TEMPLATE="alpaca"

python -m eval_vllm.driver \
    --data_file data/fresh_gaokao_math_2023.json \
    --save_dir $SAVE_DIR \
    --model_name_or_path $MODEL_PATH \
    --batch_size $BATCH_SIZE \
    --tensor_parallel_size $NUM_GPUS \
    --prompt_template $PROMPT_TEMPLATE