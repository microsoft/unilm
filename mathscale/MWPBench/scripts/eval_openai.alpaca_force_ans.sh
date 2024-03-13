OPENAI_MODEL=$1
NUM_THREADS=10
PROMPT_TEMPLATE="alpaca_force_ans"

python -m eval_openai.driver \
    --openai_model $OPENAI_MODEL \
    --num_threads $NUM_THREADS \
    --prompt_template $PROMPT_TEMPLATE \
    --verbose