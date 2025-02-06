
# Ground-truth test case inputs
python scripts/apps/solution_run_outputs_local.py \
  --completion_file "${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.json" \
  --output_file "${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.run_outputs.json" \
  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"

python scripts/apps/extract_pseudo_outputs_as_label.py \
  --input_file ${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.run_outputs.json \
  --output_file ${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_test_cases.json \
  --use_sc

# ================================================================
# Synthetic test case inputs

python scripts/apps/solution_run_pseudo_outputs_local.py \
  --completion_file "${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.json" \
  --output_file "${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_input_output.json" \
  --num_workers 64 --completion_test_field "input_output" \
  --pseudo_test_case ${DATA_PREFIX_PATH}/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only_combine.outputs.gpt4o.n1.tem0.0.json_obj.json

# =================================================

# Synthetic test cases inputs with GPT-4o for output
python scripts/apps/pseudo_test_cases/combine_gpt_raw_requests.py \
  --input_file ../msranlpintern/share/xcode_4o_oss_apps_test_inputs_v1_gpt_inputs.shuf.jsonl \
  --raw_output_file ../msranlpintern/share/xcode_4o_oss_apps_test_inputs_v1_gpt_inputs.shuf.bing.x5.jsonl_out \
  --output_file ../msranlpintern/share/xcode_4o_oss_apps_test_inputs_v1.shuf.combine.json

python scripts/apps/solution_run_outputs_local.py \
  --completion_file /mnt/fangkai_blob/share/xcode_4o_oss_apps_test_inputs_v1.shuf.combine.json \
  --output_file  /mnt/fangkai_blob/share/xcode_4o_oss_apps_test_inputs_v1.shuf.combine.4o_run_outputs.json \
  --num_workers 64 --id_field problem_id --test_case_field input_output

python scripts/apps/pseudo_test_cases/collect_pseudo_outputs.py \
  --pseudo_test_case_file ${DATA_PREFIX_PATH}/xcode_4o_oss_apps_test_inputs_v1.shuf.combine.4o_run_outputs.json \
  --output_file ${DATA_PREFIX_PATH}/xcode_4o_oss_apps_test_inputs_v1.shuf.combine.4o_ps_test_cases.sc_and_non_sc.json \
  --test_case_field "input_output"
#19795
#0
#15
#9.866784541550897
#9.013993432685021
#Counter()

# Non-sc-version
python vllm_inference.py fp16_bfloat16=True seed=42 split_id={split_id} \
  exp_name=deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42 \
  eval_sub_path=checkpoint-100 -cp conf/api/vllm/apps/deepseek_coder/r2c -cn general_combine_train_v2_0_4o_non_sc


python scripts/apps/construct_prefer_pair_soft.py \
  --input_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-apps-xcode-combine-4o-ps-tests/checkpoint-100/train.0shot.tem1.0.n10.*-of-32.v2.1.s42.json" \
  --output_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-apps-xcode-combine-4o-ps-tests/checkpoint-100/train.0shot.tem1.0.n10.v2.1.s42.prefer_pair.low0.5.m6.json" \
  --response_field response --test_case_field input_output_non_sc --pass_case_margin 6 --pass_case_lower_bound 0.5

# SC-version
python scripts/apps/solution_run_outputs_local.py \
  --completion_file "/mnt/fangkai_blob/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-apps-xcode-combine-4o-ps-tests/checkpoint-100/train.0shot.tem1.0.n10.*-of-32.v2.1.s42.json" \
  --output_file  "/mnt/fangkai_blob/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-apps-xcode-combine-4o-ps-tests/checkpoint-100/train.0shot.tem1.0.n10.v2.1.s42.run_outputs.json" \
  --num_workers 64 --id_field id --test_case_field input_output
# bash scripts/apps/pseudo_test_cases/run_outputs_local.sh split_id


python scripts/apps/pseudo_test_cases/collect_pseudo_outputs.py \
  --pseudo_test_case_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-apps-xcode-combine-4o-ps-tests/checkpoint-100/train.0shot.tem1.0.n10.v2.1.s42.run_outputs.json" \
  --output_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-apps-xcode-combine-4o-ps-tests/checkpoint-100/train.0shot.tem1.0.n10.v2.1.s42.run_outputs.prefer_pair.low0.5.m6.4o-sc.json" \
  --test_case_field "input_output" --pass_case_margin 6 --pass_case_lower_bound 0.5 --construct_prefer_pair
