

python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.*-of-256.pseudo_test_case.exec.ctr_ts_num.v1.0.json" \
    --output_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.pseudo_test_case.ctr_ts_num.prefix_pass_num.fix_low0.5_m4.0_avg.json" \
    --pass_case_lower_bound 0.5 --pass_case_margin 4 --test_case_field pseudo_input_output --reduction avg

