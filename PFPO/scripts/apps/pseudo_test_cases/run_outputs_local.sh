export OUTPUT_PREFIX_PATH=/mnt/fangkai_blob/reward_modeling/

split_id=$1

echo "Constructing process_rm sample for split $split_id"

#export exp_dir=deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-apps-xcode-combine-4o-ps-tests/checkpoint-100/
#python scripts/apps/solution_run_outputs_local.py \
#  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/${exp_dir}/train.0shot.tem1.0.n10.${split_id}-of-32.v2.1.s42.json" \
#  --output_file  "${OUTPUT_PREFIX_PATH}/experiments/${exp_dir}/train.0shot.tem1.0.n10.v2.1.${split_id}-of-32.s42.run_outputs.json" \
#  --num_workers 64 --id_field id --test_case_field input_output

# Run 4o-based prm executing
#python scripts/apps/pseudo_test_cases/prefix_fail_extract_pseudo_label.py \
#  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n5.${split_id}-of-256.v2.0.json" \
#  --output_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.${split_id}-of-256.4o_pseudo_test_case.exec.json" \
#  --num_workers 128 \
#  --pseudo_test_case /mnt/fangkai_blob/share/gpt-chat-examples-outputs/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_test_cases.json \
#  --test_case_field input_output --id_field problem_id

# 4o-DPO-Iter-0 run outputs
#python scripts/apps/solution_run_outputs_local.py \
#  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.${split_id}-of-32.v2.1.json" \
#  --output_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.${split_id}-of-32.v2.1.run_outputs.json" \
#  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"

# 4o-DPO-Iter-0 run solutions on previous 4o-synthesized test cases (APPs only)
#python scripts/apps/solution_fail_extract_pseudo_label.py \
#    --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.*-of-32.v2.1.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.v2.1.4o_pseudo_test_cases.exec.v1.0.json \
#    --pseudo_test_case /mnt/fangkai_blob/share/gpt-chat-examples-outputs/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_test_cases.json --num_workers 128

# 4o-DPO-iter-1 run outputs
#python scripts/apps/solution_run_outputs_local.py \
#  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.${split_id}-of-32.v2.1.s42.json" \
#  --output_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.${split_id}-of-32.v2.1.s42.run_outputs.json" \
#  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"

# 4o-DPO-Iter-1 run solutions on previous 4o-synthesized test cases (APPs only)
python scripts/apps/solution_fail_extract_pseudo_label.py \
    --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.apps_only.${split_id}-of-9.v2.1.s42.json" \
    --output_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.apps_only.v2.1.${split_id}-of-9.s42.4o_pseudo_test_cases.exec.v1.0.json" \
    --pseudo_test_case /mnt/fangkai_blob/share/gpt-chat-examples-outputs/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_test_cases.json --num_workers 128

