OUTPUT_PREFIX_PATH=/mnt/fangkai_blob/reward_modeling/

split_id=$1

python scripts/apps/pseudo_test_cases/prefix_fail_extract_pseudo_label_align_ts_num.py \
    --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n5.${split_id}-of-256.v2.0.json" \
    --output_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.${split_id}-of-256.pseudo_test_case.exec.ctr_ts_num.v1.0.json" \
    --num_workers 64 \
    --pseudo_test_cases ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.json

