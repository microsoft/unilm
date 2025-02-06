### Synthesis Test Case Inputs

Prepare inputs for prompting GPT-4o:

```bash
python scripts/apps/pp_test_case_gen_inputs_v2.0.py \
  --split train --prompt_file prompts/apps/test_input_gen_2shot_v2.0.txt \
  --output outputs/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.0.inputs.jsonl
    
python scripts/apps/pp_test_case_gen_inputs_v2.0.py \
  --split train --prompt_file prompts/apps/test_input_gen_2shot_v2.1.txt \
  --output outputs/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only.inputs.jsonl --function_only
```

Extract the corresponding outputs from GPT-4o and combine them to obtain the following file:

`../msranlpintern/share/gpt-chat-examples-outputs/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only_combine.outputs.gpt4o.n1.tem0.0.json_obj.json`

For post-processing to align the format with oss data:

```bash
python scripts/apps/pseudo_test_cases/combine_pseudo_test_inputs.py \
    --output_file ../msranlpintern/share/dataset/magicoder/apps-train.4o-test-func.json \
    --pseudo_test_case ../msranlpintern/share/gpt-chat-examples-outputs/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only_combine.outputs.gpt4o.n1.tem0.0.json_obj.json
```

For directly processing on APPs:

```bash
python scripts/apps/solution_run_pseudo_outputs_local.py \
    --completion_file "../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.?-of-8.v2.0.json" \
    --output_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.json \
    --pseudo_test_case outputs/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only_combine.outputs.gpt4o.n1.tem0.0.json_obj.json 
    
    
python scripts/apps/solution_run_pseudo_outputs_local.py \
    --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.?-of-4.v2.0.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_test_cases.v1.0.azure.json \
    --pseudo_test_case ${DATA_PREFIX_PATH}/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only_combine.outputs.gpt4o.n1.tem0.0.json_obj.json --num_workers 128


python scripts/apps/solution_run_pseudo_outputs_local.py \
  --completion_file "${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.json" \
  --output_file "${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_input_output.json" \
  --num_workers 64 --completion_test_field "input_output" \
  --pseudo_test_case ${DATA_PREFIX_PATH}/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only_combine.outputs.gpt4o.n1.tem0.0.json_obj.json
```

Collect pseudo outputs and construct DPO training pairs

```bash
python scripts/apps/pseudo_test_cases/collect_pseudo_outputs.py \
    --pseudo_test_case_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_test_cases.v1.0.azure.json \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.clean.dpo_m6_low0.5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 
 
```

Either, run execution again on existing pseudo test cases.

First, extract the pseudo label:
```bash
python scripts/apps/extract_pseudo_outputs_as_label.py \
  --input_file ${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_input_output.json \
  --output_file ${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_test_cases.json \
  --use_sc --problem_id_field problem_id --test_case_field pseudo_test_cases 
#Unaligned: 0
#Total number of pseudo test cases: 3901
#Average number of test cases: 36333 / 3901 = 9.313765701102282
```

Secondly, run execution on pseudo test cases:
```bash
python scripts/apps/solution_fail_extract_pseudo_label.py \
    --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.?-of-4.v2.0.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.4o_pseudo_test_cases.v1.0.azure.json \
    --pseudo_test_case ${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_test_cases.json --num_workers 128
#Missing: 0.0002563445270443476
#Correct: 0.08203024865419123
#Correct at k: 0.1579082286593181
#Counter({False: 36311, True: 2689})
```

Construct (DPO) preference pair:
```bash
python scripts/apps/construct_prefer_pair.py \
  --input_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.4o_pseudo_test_cases.v1.0.azure.json \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.4o_pseudo_test_cases.prefer_pair.json \
  --test_case_field pseudo_input_output
# 3901 9165 2.3493975903614457

# Soft version
python scripts/apps/construct_prefer_pair_soft.py \
  --input_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.4o_pseudo_test_cases.v1.0.azure.json \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.4o_pseudo_test_cases.dpo_m4_low0.5.json \
  --test_case_field pseudo_input_output --pass_case_margin 4 --pass_case_lower_bound 0.5 
# 3901 12851 3.294283517046911
```

Run process-DPO prefix execution
```bash
python scripts/apps/pseudo_test_cases/prefix_fail_extract_pseudo_label.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n5.${split_id}-of-256.v2.0.json" \
  --output_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.${split_id}-of-256.4o_pseudo_test_case.exec.json" \
  --num_workers 64 \
  --pseudo_test_case ../msranlpintern/share/gpt-chat-examples-outputs/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_test_cases.json \
  --test_case_field input_output --id_field problem_id
```

Construct Process-DPO preference pair
```bash
python scripts/apps/prm/construct_process_rm_sample_fix.py \
  --input_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.*-of-256.4o_pseudo_test_case.exec.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.4o_pseudo_test_case.prm_prefer_pair.json \
  --pass_case_margin 4 --pass_case_lower_bound 0.5 --test_case_field pseudo_input_output --num_workers 24 --reduction avg 
#Missing: 0
#Missing test cases: 0
#Counter({0: 1440638, 10: 108458, 1: 80861, 2: 34440, 3: 24141, 4: 18476, 5: 15235, 9: 14017, 6: 13494, 7: 12309, 8: 11546})
#Processed 354723 prefixes.
#Averaged 1.0 prefixes per problem.
#Processed 3900 problems.
```
---------------------------------------------------------------

Based on DPO-Iter-0 model to construct new test cases on magicoder

Run outputs on newly generated oss-apps-combine data
```bash
python scripts/apps/solution_run_outputs_local.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.{split_id}-of-32.v2.1.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.{split_id}-of-32.v2.1.run_outputs.json \
  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"
```

Combine oss outputs with test case inputs to obtain self-consistency labels, and get dpo training pairs.
```bash
python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_mp.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.*-of-32.v2.1.run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.v2.1.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5
#11925
#11925
#117006
#0
#10.004442649434571
#9.823396226415094
#Counter({10: 73857, 0: 23639, 9: 3901, 1: 3222, 8: 2589, 5: 2251, 2: 2103, 7: 1976, 4: 1865, 6: 1865, 3: 1852, 11: 105, 20: 25})
```
We need to reuse the pre-synthesized pseudo test cases for APPs dataset for training.
```bash
python scripts/apps/solution_fail_extract_pseudo_label.py \
    --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.*-of-32.v2.1.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.v2.1.4o_pseudo_test_cases.exec.v1.0.json \
    --pseudo_test_case ${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_test_cases.json --num_workers 128
```

Construct (DPO) preference pair:
```bash
# Soft version
python scripts/apps/construct_prefer_pair_soft.py \
  --input_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.v2.1.4o_pseudo_test_cases.exec.v1.0.json \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_4o_ps_test_case.dpo.H100.dp8.v1.0.s42/oss-instruct-apps-train/checkpoint-100/train.0shot.tem1.0.n10.v2.1.apps_only.4o_pseudo_test_cases.dpo_m4_low0.5.json \
  --test_case_field pseudo_input_output --pass_case_margin 4 --pass_case_lower_bound 0.5 
3901
3901 11364 2.913099205331966
```

---------------------------------------------------------------

Based on DPO-Iter-1 model to construct new test cases on magicoder and xcode

```bash
python scripts/apps/solution_run_outputs_local.py \
  --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.{split_id}-of-32.v2.1.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.{split_id}-of-32.v2.1.run_outputs.json \
  --num_workers 64 --id_field "problem_id" --test_case_field "input_output"
```

Combine oss outputs with test case inputs to obtain self-consistency labels, and get dpo training pairs.
```bash
python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_mp.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.*-of-32.v2.1.s42.run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.v2.1.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5
# we have skipped split id 20 since it uses too much time.
#18608
#18608
#193834
#0
#10.003605287755375
#9.80116079105761
#Counter({10: 99051, 0: 37603, 1: 8459, 9: 7563, 2: 5656, 8: 5528, 3: 4714, 5: 4698, 7: 4306, 4: 4260, 6: 4070, 11: 136, 20: 34, 12: 2})

python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs_mp.py \
    --pseudo_test_case_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.*-of-32.v2.1.s42.run_outputs.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.v2.1.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.all.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5
# Added the skipped split id 20
#19051
#19051
#197647
#0
#10.003557222779161
#9.800902839745945
#Counter({10: 102049, 0: 38432, 1: 8511, 9: 7700, 2: 5688, 8: 5591, 3: 4783, 5: 4776, 7: 4358, 4: 4329, 6: 4119, 11: 136, 20: 34, 12: 4})
```

We need to reuse the pre-synthesized pseudo test cases for APPs dataset for training.
```bash
python scripts/apps/solution_fail_extract_pseudo_label.py \
    --completion_file "${OUTPUT_PREFIX_PATH}/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.*-of-32.v2.1.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.apps_only.4o_pseudo_test_cases.exec.v1.0.json \
    --pseudo_test_case ${DATA_PREFIX_PATH}/apps-train-sub-train-eval-outputs-v2.1-gpt4o-tem0.0-seq4k-pipe-format.pseudo_test_cases.json --num_workers 128
```

Construct (DPO) preference pair:
```bash
# Soft version
python scripts/apps/construct_prefer_pair_soft.py \
  --input_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300//train.0shot.tem1.0.n10.apps_only.v2.1.*-of-9.s42.4o_pseudo_test_cases.exec.v1.0.json" \
  --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps-4o.mc-self.iter1.dpo.H100.dp32.v1.0.s42/oss-apps-xcode-combine/checkpoint-300/train.0shot.tem1.0.n10.v2.1.apps_only.4o_pseudo_test_cases.dpo_m4_low0.5.json \
  --test_case_field pseudo_input_output --pass_case_margin 4 --pass_case_lower_bound 0.5 

```
