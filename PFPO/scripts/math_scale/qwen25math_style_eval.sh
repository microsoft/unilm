split_id=$1

#exp_name=qwen2.5-math.7b.base.math.long_cot.sft.MI300x.dp16.v5.0.1.s42
exp_name=qwen2.5-math.7b.base.numina-level5-refine.long_cot.pattern-dpo.mi300x.dp32.iter0.v1.8.s42

#bash scripts/math_scale/merge_mwpbench_predictions.sh /mnt/fangkai_blob/reward_modeling/experiments/${exp_name} train.wo.math.gsm.2k 2.0 "${split_id}"

#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/reward_modeling/experiments/${exp_name}/mwpbench/checkpoint-${split_id}/train.wo.math.gsm.2k.test.v2.0.0shot.n1.tem0.0.p1.0.0-of-1.json" \
#  --num_workers 1
python scripts/math_scale/qwen25math_style_eval_v2.0.py \
  --input_file "/mnt/fangkai_blob/reward_modeling/experiments/${exp_name}/math500/checkpoint-${split_id}/math.test.v2.0.0shot.n1.tem0.0.p1.0.0-of-1.json" \
  --num_workers 1 --source_field subject


#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/reward_modeling/experiments/${exp_name}/checkpoint-1900/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.${split_id}-of-128.s0.json" \
#  --num_workers 64 --source_field source

#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/reward_modeling/experiments/${exp_name}/checkpoint-1900/eurus-2-rl-data/train.numina.no_box_inst/cot.n4.tem1.0.p0.7.v2.0.${split_id}-of-128.s1.json" \
#  --num_workers 64 --source_field source
#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/reward_modeling/experiments/${exp_name}/checkpoint-1900/eurus-2-rl-data/train.numina.no_box_inst/cot.n8.tem1.0.p0.7.v2.0.s0.sympy_eval.r1_all_wrong.n8.tem1.0.p0.8.v2.0.${split_id}-of-128.s3.json" \
#  --num_workers 64 --source_field source

#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/reward_modeling/experiments/${exp_name}/checkpoint-1900/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.${split_id}-of-128.s0.json" \
#  --num_workers 64 --source_field source \
#  --label_file "/mnt/fangkai_blob/share/models/Qwen2.5-Math-72B-Instruct/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.s0.sympy_preprocess.refined.freq0.0.v1.0.json" \
#  --output_file "/mnt/fangkai_blob/reward_modeling/experiments/${exp_name}/checkpoint-1900/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.${split_id}-of-128.s0.refined.qwen25-math-72b-maj8-freq0.0.v1.0.sympy_eval.json"

#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/reward_modeling/experiments/${exp_name}/checkpoint-1900/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.s0.refined.qwen25-math-72b-maj8-freq0.0.v1.0.sympy_eval.alter-iter1.n3.tem1.0.p0.7.v1.1.${split_id}-of-128.s0.json" \
#  --num_workers 64 --source_field source

#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/share/models/Qwen2.5-Math-72B-Instruct/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.s0.sympy_preprocess.refined.freq0.0.v1.0.${split_id}-of-64.json" \
#  --num_workers 64 --source_field source
#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/share/models/Qwen2.5-Math-7B-Instruct/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.${split_id}-of-64.s0.json" \
#  --num_workers 64 --source_field source
#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/share/models/Qwen2.5-Math-7B-Instruct/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.${split_id}-of-64.s0.json" \
#  --num_workers 64 --source_field source \
#  --label_file "/mnt/fangkai_blob/share/models/Qwen2.5-Math-72B-Instruct/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.s0.sympy_preprocess.refined.freq0.0.v1.0.json" \
#  --output_file "/mnt/fangkai_blob/share/models/Qwen2.5-Math-7B-Instruct/numina.aops_aime_oly.level_5_v1.0/cot.de_con.n8.tem1.0.p0.7.v2.0.${split_id}-of-64.s0.refined.freq0.0.v1.0.sympy_eval.json"

#python scripts/math_scale/qwen25math_style_eval_v2.0.py \
#  --input_file "/mnt/fangkai_blob/share/models/Llama-3.3-70B-Instruct/numina/math-difficulty/math.cot.de_con.16k.n4.tem1.0.p0.85.v5.0.s0.sympy_eval.iter1-alter.16k.n1.tem1.0.p0.85.${split_id}-of-16.v1.0.s0.json" \
#  --num_workers 64 --source_field data_topic