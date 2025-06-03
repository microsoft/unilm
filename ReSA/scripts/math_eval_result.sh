tasks=("minerva_math" "gaokao2023en" "olympiadbench" "aime24" "amc23")
# tasks=("gsm8k" "math" "svamp" "asdiv" "mawps" "carp_en" "tabmwp" "minerva_math" "gaokao2023en" "olympiadbench" "college_math" "aime24" "amc23")

output_folder=$1
result_file=$2

prompt_type="r1"
eval_num=-1

export TQDM_DISABLE=1
for task in "${tasks[@]}"; do
    python scripts/math_eval_result_length.py \
        --data_names $task \
        --prompt_type $prompt_type \
        --result_file ${output_folder}/${task}/${result_file} \
        --eval_num $eval_num
done