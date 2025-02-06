export OUTPUT_PREFIX_PATH=../msranlpintern/reward_modeling/

python scripts/apps/pseudo_test_cases/clean_xcode_4o_test_inputs_data.py \
    --data_file ../msranlpintern/share/xCodeEval/problem_descriptions.jsonl \
    --test_case_file ../gpt_crawler/outputs/xcode_test_case_inputs_gen_v2.1.4o.tem0.0.n1.json_obj.jsonl \
    --output_file ../msranlpintern/share/xCodeEval/xcode_train_4o_test_inputs_v1.json \
    --test_file "../msranlpintern/share/xCodeEval/**/test/*.jsonl" \
    --val_file "../msranlpintern/share/xCodeEval/**/validation/*.jsonl"
#Test cases: 7588 / 7635
#6163
#No test: 1372

# Run outputs on newly generated data
python scripts/apps/solution_run_outputs_local.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/xcode-train/checkpoint-800/train.0shot.tem1.0.n10.*-of-16.v2.0.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/xcode-train/checkpoint-800/train.0shot.tem1.0.n10.*-of-16.v2.0.run_outputs.json \
  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"

# Combine outputs with test case inputs to obtain self-consistency labels, and get dpo training pairs.
python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/xcode-train/checkpoint-800/train.0shot.tem1.0.n10.*-of-16.v2.0.run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/xcode-train/checkpoint-800/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5
#5844
#79260
#0
#10.0
#9.7298083504449
#Counter({0: 15165, 10: 14577, 1: 5613, 9: 3884, 2: 3674, 8: 3100, 3: 2896, 4: 2561, 7: 2376, 5: 2350, 6: 2244})


# Run outputs on newly generated oss-apps-combine data
python scripts/apps/solution_run_outputs_local.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-instruct-apps-train/checkpoint-800/train.0shot.tem1.0.n10.{split_id}-of-32.v2.0.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-instruct-apps-train/checkpoint-800/train.0shot.tem1.0.n10.{split_id}-of-32.v2.0.run_outputs.json \
  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"

# Combine oss outputs with test case inputs to obtain self-consistency labels, and get dpo training pairs.
python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-instruct-apps-train/checkpoint-800/train.0shot.tem1.0.n10.*-of-32.v2.0.run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-instruct-apps-train/checkpoint-800/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5
#14107
#106950
#1
#10.004038772213248
#9.794782731977033
#Counter({10: 96073, 0: 20843, 9: 4661, 1: 3399, 8: 3067, 5: 2755, 2: 2179, 7: 2116, 6: 2004, 3: 1940, 4: 1870, 11: 142, 20: 21})


# Sample steps for xcode
python scripts/apps/prm/sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/xcode-train/checkpoint-800/train.0shot.tem1.0.n10.*-of-16.v2.0.json" \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --sample_over_p 10 \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/xcode-train/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.json
# Total number of samples: 61630

# Sample steps for new oss-apps
python scripts/apps/prm/sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-instruct-apps-train/checkpoint-800/train.0shot.tem1.0.n10.*-of-32.v2.0.json" \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --sample_over_p 10 \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-instruct-apps-train/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.json
# Total number of samples: 247580

python scripts/apps/pseudo_test_cases/oss_combine_prefix_fail_extract_pseudo_label.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.{split_id}-of-32.v2.0.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.{split_id}-of-32.pseudo_input_output.exec.json \
  --num_workers 128 --pseudo_test_cases ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5_rm_large_meta.json


# Construct prm pairs
python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.*-of-32.pseudo_input_output.exec.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.pseudo_input_output.prefix_pass_num.fix.json \
    --pass_case_lower_bound 0.5 --pass_case_margin 4 --test_case_field pseudo_input_output --reduction avg --test_case_field input_output

python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.*-of-32.pseudo_input_output.exec.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.pseudo_input_output.prefix_pass_num.fix.json \
    --pass_case_lower_bound 0.6 --pass_case_margin 6 --test_case_field pseudo_input_output --reduction avg --test_case_field input_output

# Run outputs on apps-magicoder-xcode-combine newly generated data w/ n=64
python scripts/apps/solution_run_outputs_local.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n8.*-of-32.v2.0.s[0-8].json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.*-of-16.v2.0.run_outputs.json \
  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"


# Combine outputs with test case inputs to obtain self-consistency labels, and get dpo training pairs.
python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_mp.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n8.*-of-32.v2.0.s[0-8].run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n64.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5

# Just check the outputs on the previously generated data
python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_takes_extra.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n8.*-of-32.v2.0.s[0-8].run_outputs.json" \
    --completion_file  "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.pseudo_input_output_by_n64.v1.0.dpo_m6_low0.5_min5.json \
    --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5
#21790
#188616
#0
#10.003557222779161
#9.875906379072969
#Counter({10: 109418, 0: 55905, 1: 9618, 9: 8233, 2: 6063, 8: 5869, 5: 4976, 3: 4861, 4: 4392, 7: 4323, 6: 4073, 11: 146, 20: 23})  # TODO: Why there is 20 and 11????

#21790
#188616
#0
#10.003557222779161
#9.875906379072969
#Counter({10: 109418, 0: 55905, 1: 9618, 9: 8233, 2: 6063, 8: 5869, 5: 4976, 3: 4861, 4: 4392, 7: 4323, 6: 4073, 11: 146, 20: 23})

#python scripts/apps/pseudo_test_cases/oss_combine_prefix_fail_extract_pseudo_label.py \
#  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.{split_id}-of-32.v2.0.json" \
#  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.{split_id}-of-32.pseudo_input_output.exec.json \
#  --num_workers 128 --pseudo_test_cases ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-apps-xcode-combine/checkpoint-800/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5_rm_large_meta.json
#

# Starting from model w/ magicoder & apps - pdpo, repeat the process again.
# Run outputs on newly generated data
exp_dir=deepseek-coder-v1.5-ins.7b.r2c.sft_ps_test_case.iter2.pdpo.V100.tp8dp32.v1.3.s42/oss-apps-xcode-combine/checkpoint-300/

python scripts/apps/solution_run_outputs_local.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.{split_id}-of-32.v2.0.s[0-8].json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.{split_id}-of-32.v2.0.run_outputs.json \
  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"

# Sample steps for xcode
python scripts/apps/prm/sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.*-of-32.v2.0.s[0-8].json" \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --sample_over_p 10 \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample10_per.json
# Total number of samples: 61630

# Combine outputs with test case inputs to obtain self-consistency labels, and get dpo training pairs.
python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_mp.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.*-of-32.s[0-9].v2.0.run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5
#22303
#22303
#8588574
#0
#10.003557222779161
#9.884141146930906
#Counter({10: 767611, 0: 324963, 1: 57875, 9: 53845, 2: 37125, 8: 36508, 5: 31254, 3: 30650, 7: 28261, 4: 27218, 6: 26200, 11: 990, 20: 188})

# Run completion outputs
python scripts/apps/pseudo_test_cases/oss_combine_prefix_fail_extract_pseudo_label.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.glo-{global_split_id}-of-8.loc-{split_id}-of-64.v2.0.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.glo-{global_split_id}-of-8.loc-{split_id}-of-64.pseudo_input_output.exec.json \
  --num_workers 128 --pseudo_test_cases ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5sc_test_cases.json


# Construct prm pairs
python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.glo-*-of-8.loc-*-of-64.pseudo_input_output.exec.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.pseudo_input_output.prefix_pass_num.fix.json \
    --pass_case_lower_bound 0.5 --pass_case_margin 4 --test_case_field pseudo_input_output --reduction avg --test_case_field input_output
#Missing: 890
#Missing test cases: 0
#Counter({0: 268084, 10: 15225, 1: 7396, 9: 2466, 2: 2270, 3: 1795, 8: 1696, 5: 1385, 4: 1371, 6: 1290, 7: 1207})
#Processed 101395 prefixes.
#Averaged 0.9912988219191475 prefixes per problem.
python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n10.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.glo-*-of-8.loc-*-of-64.pseudo_input_output.exec.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample10_per.completion.tem1.0.n3.pseudo_input_output.prefix_pass_num.fix.json \
    --pass_case_lower_bound 0.3 --pass_case_margin 3 --test_case_field pseudo_input_output --reduction avg --test_case_field input_output
#Missing: 890
#Missing test cases: 0
#Counter({0: 268084, 10: 15225, 1: 7396, 9: 2466, 2: 2270, 3: 1795, 8: 1696, 5: 1385, 4: 1371, 6: 1290, 7: 1207})
#Processed 101395 prefixes.
#Averaged 0.9912988219191475 prefixes per problem.
#Processed 10172 problems.

exp_dir=deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.pdpo.H100.dp8.v1.2.s42/oss-apps-xcode-combine/checkpoint-500/

python scripts/apps/prm/sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.*-of-32.v2.0.s[0-8].json" \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --sample_over_p 10 \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample10_per.json
#Too large outputs: 0
#Total number of samples: 309210
python scripts/apps/prm/sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.*-of-32.v2.0.s[0-8].json" \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --sample_over_p 32 \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample32_per.json
#Too large outputs: 0
#Total number of samples: 989473

python scripts/apps/solution_run_outputs_local.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.{split_id}-of-32.v2.0.s{seed}.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.{split_id}-of-32.s{seed}.v2.0.run_outputs.json \
  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"

python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_mp_compress.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n8.*-of-32.s[0-9].v2.0.run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.pseudo_input_output.v1.0.cp.dpo_m4_low0.3_min5_p$p.json \
    --construct_prefer_pair --pass_case_margin 4 --pass_case_lower_bound 0.3 --min_success_test_num 5 --top_p $p
#p=0.0
#21662
#21662
#608162
#0
#10.003557222779161
#9.874849967685348
#Counter({10: 679443, 0: 368401, 1: 61978, 9: 52376, 2: 39749, 8: 36629, 3: 31688, 5: 31673, 4: 28822, 7: 28171, 6: 26357, 11: 909, 20: 170, 12: 2})
#p=0.2
#18373
#18373
#585808
#0
#10.003557222779161
#9.93577532248408
#Counter({10: 668678, 0: 205613, 9: 50259, 1: 46523, 8: 34864, 2: 33257, 5: 29338, 3: 28297, 7: 26725, 4: 26387, 6: 24873, 11: 886, 20: 170, 12: 2})
#p=0.4
#14583
#14583
#506479
#0
#10.003557222779161
#9.954878968662142
#Counter({10: 634922, 0: 95150, 9: 41702, 8: 28477, 1: 22680, 5: 20957, 7: 20445, 6: 18610, 2: 17527, 4: 16296, 3: 15538, 11: 838, 20: 170})
#=0.5
#12903
#12903
#452734
#0
#10.003557222779161
#9.960861815081763
#Counter({10: 609930, 0: 62866, 9: 36059, 8: 23682, 7: 16213, 5: 15332, 6: 14783, 1: 14003, 2: 11235, 4: 10667, 3: 10087, 11: 791, 20: 144})


# Run completion outputs
python scripts/apps/pseudo_test_cases/oss_combine_prefix_fail_extract_pseudo_label.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample32_per.completion.tem1.0.n3.glo-{global_split_id}-of-16.loc-{split_id}-of-64.v2.0.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample32_per.completion.tem1.0.n3.glo-{global_split_id}-of-16.loc-{split_id}-of-64.v2.0.pseudo_input_output.exec.json \
  --num_workers 128 --pseudo_test_cases ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.pseudo_input_output.v1.0.cp.dpo_m4_low0.3_min5_p0.0.sc_test_cases.json

# p=0.0
python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample32_per.completion.tem1.0.n3.glo-*-of-16.loc-*-of-64.v2.0.pseudo_input_output.exec.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample32_per.completion.tem1.0.n3.pseudo_input_output.prefix_pass_num.fix.json \
    --pass_case_lower_bound 0.5 --pass_case_margin 4 --test_case_field pseudo_input_output --reduction avg --test_case_field input_output
#Missing: 4235
#Missing test cases: 0
#Counter({0: 1492601, 10: 211726, 1: 90982, 2: 45672, 9: 42604, 3: 34232, 8: 32685, 4: 29914, 5: 27977, 7: 26170, 6: 25720})
#Processed 686761 prefixes.
#Averaged 0.9938711656796856 prefixes per problem.
#Processed 21533 problems.

# p=0.2
python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample32_per.completion.tem1.0.n3.glo-*-of-16.loc-*-of-64.v2.0.pseudo_input_output.p0.2.exec.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/$exp_dir/train.0shot.tem1.0.n64.v2.0.prefix.upper0.8.r0.3.sample32_per.completion.tem1.0.n3.pseudo_input_output.prefix_pass_num.p0.2.fix.json \
    --pass_case_lower_bound 0.5 --pass_case_margin 4 --test_case_field pseudo_input_output --reduction avg --test_case_field input_output
#Missing: 2073
#Missing test cases: 0
#Counter({0: 952288, 10: 159840, 1: 54560, 9: 29678, 2: 27823, 8: 23059, 3: 21733, 4: 20014, 5: 18776, 7: 17984, 6: 17690})
#Processed 447815 prefixes.
#Averaged 0.995392186499751 prefixes per problem.
#Processed 14028 problems.