OPENAI_MODEL=$1
SAVE_DIR=results/fresh_gaokao_math_2023.$OPENAI_MODEL
NUM_THREADS=10
PROMPT_TEMPLATE="alpaca_force_ans"

python -m eval_openai.driver \
    --data_file data/fresh_gaokao_math_2023.json \
    --save_dir $SAVE_DIR \
    --openai_model $OPENAI_MODEL \
    --num_threads $NUM_THREADS \
    --prompt_template $PROMPT_TEMPLATE \
    --verbose