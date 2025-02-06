#python scripts/apps/solution_run_outputs_local.py --completion_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.0shot.tem1.0.n10.?-of-8.v2.0.json" --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.0shot.tem1.0.n10.v2.0.run_outputs.json --num_workers 24 --id_field "problem_id" --test_case_field "input_output"
python scripts/apps/solution_run_outputs_local.py --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.{split_id}-of-32.v2.0.json" --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.{split_id}-of-32.run_outputs.json --num_workers 64 --id_field "problem_id" --test_case_field "input_output"

python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs.py \
    --pseudo_test_case_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.*-of-32.run_outputs.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5

# Iter - 1: PRM
# Sample prefix
python scripts/apps/prm/sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.*-of-32.v2.0.json" \
    --upper_step_ratio 0.8 --sample_ratio 0.3 \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.prefix.upper0.8.r0.3.v2.0.json

python scripts/apps/prm/sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.*-of-32.v2.0.json" \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --sample_over_p 20 \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample20_per.json


# Execute the prefix completions
python scripts/apps/pseudo_test_cases/oss_combine_prefix_fail_extract_pseudo_label.py \
    --completion_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.tem1.0.n10.prefix.upper0.8.r0.3.sample20_per.completion.tem1.0.n3.*-of-256.v2.0.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.tem1.0.n10.prefix.upper0.8.r0.3.sample20_per.completion.tem1.0.n3.pseudo_input_output.exec.json \
    --num_workers 24 \
    --pseudo_test_cases ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json

# Construct preference pairs
python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "$OUTPUT_PATH_PREFIX/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.tem1.0.n10.prefix.upper0.8.r0.3.sample20_per.completion.tem1.0.n3.pseudo_input_output.exec.*-of-256.json" \
    --output_file $OUTPUT_PATH_PREFIX/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.tem1.0.n10.prefix.upper0.8.r0.3.sample20_per.completion.tem1.0.n3.pseudo_input_output.prefix_pass_num.fix.json \
    --pass_case_lower_bound 0.5 --pass_case_margin 4 --test_case_field pseudo_input_output --reduction avg --test_case_field input_output

