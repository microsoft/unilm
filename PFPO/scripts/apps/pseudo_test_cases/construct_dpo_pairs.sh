export OUTPUT_PREFIX_PATH=/mnt/fangkai_blob/reward_modeling/
#export exp_dir=deepseek-coder-v1.5-ins.7b.r2c.sft_ps_test_case.iter2.pdpo.V100.tp8dp32.v1.3.s42/oss-apps-xcode-combine/checkpoint-300/
export exp_dir=deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.pdpo.H100.dp8.v1.2.s42/oss-apps-xcode-combine/checkpoint-500/

p=$1

echo "Constructing process_rm sample for split 0"
echo "p" $p

# Just check the outputs on the previously generated data
#python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_takes_extra.py \
#    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n8.*-of-32.v2.0.s[0-8].run_outputs.json" \
#    --completion_file  "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.run_outputs.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.pseudo_input_output_by_n64.v1.0.dpo_m6_low0.5_min5_p$p.json \
#    --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5 --top_p $p


# Combine outputs with test case inputs to obtain self-consistency labels, and get dpo training pairs.
#python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_mp.py \
#    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.*-of-32.s[0-9].v2.0.run_outputs.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5_p$p.json \
#    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5 --top_p $p


## Combine outputs with test case inputs to obtain self-consistency labels, and get dpo training pairs.
#python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_mp_compress.py \
#    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.*-of-32.s[0-9].v2.0.run_outputs.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.pseudo_input_output.v1.0.cp.dpo_m4_low0.3_min5_p$p.json \
#    --construct_prefer_pair --pass_case_margin 4 --pass_case_lower_bound 0.3 --min_success_test_num 5 --top_p $p


# Combine outputs with test case inputs to obtain self-consistency labels, and get dpo training pairs.
python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_mp_compress.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.*-of-32.s[0-9].v2.0.run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.pseudo_input_output.v1.0.cp.dpo_m4_low0.3_min5_p$p.json \
    --construct_prefer_pair --pass_case_margin 4 --pass_case_lower_bound 0.3 --min_success_test_num 5 --top_p $p