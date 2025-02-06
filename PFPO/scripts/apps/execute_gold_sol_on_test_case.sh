export OUTPUT_PATH_PREFIX="/mnt/fangkai_blob/reward_modeling/"


#python scripts/apps/execute_gold_sol_on_test_case.py \
#  --test_case_file ${OUTPUT_PATH_PREFIX}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.clean.dpo_m6_low0.5.json \
#  --test_case_field "pseudo_test_cases" --test_case_id_field "id" \
#  --output_file ${OUTPUT_PATH_PREFIX}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.clean.dpo_m6_low0.5.gd_sol_exec.json \
#  --num_workers 128
#Total number of items before: 5000
#4500
#Missing solutions: 0
#Missing test cases: 264
#Total number of items: 4236
#Missing: 0 / 4236 = 0.0
#Correct: 1831 / 4236 = 0.43224740321057603
#Micro Correct: 26256 / 41889 = 0.6267993984100838

# deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42
#python scripts/apps/execute_gold_sol_on_test_case.py \
#  --test_case_file ${OUTPUT_PATH_PREFIX}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
#  --test_case_field "input_output" --test_case_id_field "id" \
#  --output_file ${OUTPUT_PATH_PREFIX}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.gd_sol_exec.json \
#  --num_workers 128
#Total number of items before: 5000
#4500
#Missing solutions: 0
#Missing test cases: 377
#Total number of items: 4123
#Missing: 0 / 4123 = 0.0
#Correct: 2066 / 4123 = 0.5010914382731021
#Micro Correct: 26954 / 40349 = 0.6680215123051376


#deepseek-coder-v1.5-ins.7b.r2c.sft_ps_test_case.iter2.dpo.H100.dp8.v1.0.s42
#python scripts/apps/execute_gold_sol_on_test_case.py \
#  --test_case_file ${OUTPUT_PATH_PREFIX}//experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-instruct-apps-train/checkpoint-800/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
#  --test_case_field "input_output" --test_case_id_field "id" \
#  --output_file ${OUTPUT_PATH_PREFIX}//experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.iter1.dpo.A100.tp4dp16.v1.2.s42/oss-instruct-apps-train/checkpoint-800/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.gd_sol_exec.json \
#  --num_workers 128
#4500
#Missing solutions: 0
#Missing test cases: 331
#Total number of items: 4169
#Missing: 0 / 4169 = 0.0
#Correct: 2077 / 4169 = 0.49820100743583595
#Micro Correct: 27217 / 40775 = 0.6674923359901901


#deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42
#python scripts/apps/execute_gold_sol_on_test_case.py \
#  --test_case_file ${OUTPUT_PATH_PREFIX}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.4o_pseudo_test_cases.dpo_m4_low0.5.json \
#  --test_case_field "input_output" --test_case_id_field "id" \
#  --output_file ${OUTPUT_PATH_PREFIX}//experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.4o_pseudo_test_cases.dpo_m4_low0.5.gd_sol_exec.json \
#  --num_workers 128
#4500
#Missing: 1
#Repeat: 0
#Missing solutions: 0
#Missing test cases: 600
#Total number of items: 3900
#Missing: 0 / 3900 = 0.0
#Correct: 511 / 3900 = 0.13102564102564102
#Micro Correct: 50202 / 60914 = 0.8241455166300029

# The sc-version of 4o-based test cases
python scripts/apps/execute_gold_sol_on_test_case.py \
  --test_case_file /mnt/fangkai_blob/share/xcode_4o_oss_apps_test_inputs_v1.shuf.combine.4o_ps_test_cases.sc_and_non_sc.json \
  --test_case_field "input_output" --test_case_id_field "problem_id" \
  --output_file /mnt/fangkai_blob/share/xcode_4o_oss_apps_test_inputs_v1.shuf.combine.4o_ps_test_cases.sc_and_non_sc.gd_sol_exec.json \
  --num_workers 128
#Total number of items before: 5000
#4500
#Missing: 0
#Repeat: 0
#Missing solutions: 0
#Missing test cases: 248
#Total number of items: 4252
#Collecting results: 100%|██████████| 4252/4252 [00:54<00:00, 78.74it/s]
#Missing: 0 / 4252 = 0.0
#Correct: 2697 / 4252 = 0.6342897460018815
#Micro Correct: 32348 / 42080 = 0.7687262357414448