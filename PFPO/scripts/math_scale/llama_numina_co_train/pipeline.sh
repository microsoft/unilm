export OUTPUT_PREFIX_PATH=../msranlpintern/reward_modeling/

python scripts/math/deepseek_math_sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/830k-split-[23]-of-10/cot.de_con.[23]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.json \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --filter_all_same --sample_over_p 8

python scripts/math/deepseek_math_sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/830k-split-[01]-of-10/cot.de_con.[01]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split01.upper0.8.r0.3.sample8.filter_same.json \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --filter_all_same --sample_over_p 8


python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split01.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-512.s0.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split01.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/830k-split-[01]-of-10/cot.de_con.[01]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p
# p=0.0
# Counter({0: 941033, 1: 550298, 3: 422913, 2: 412766})
# Missing 0 items in the response data.
# Counter({0: 910219, 1: 532186, 3: 408604, 2: 398571})
# p=0.5
# Counter({3: 337436, 2: 215444, 1: 152720, 0: 122527})
# Missing 1498883 items in the response data.
# Counter({3: 325974, 2: 208001, 1: 147657, 0: 118407})
python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-512.s0.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/830k-split-[23]-of-10/cot.de_con.[23]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p
# p=0.0
# Counter({0: 1019074, 1: 593140, 3: 451050, 2: 444011})
# Missing 0 items in the response data.
# Counter({0: 916032, 1: 532522, 3: 405042, 2: 398580})
# p=0.5
# Counter({3: 360314, 2: 233018, 1: 164644, 0: 132445})
# Missing 1616854 items in the response data.
# Counter({3: 323596, 2: 209373, 1: 148006, 0: 119201})


# Sample from the dpo-trained co- models, and re-computing the process-rm labels.


# Change to vanilla iterative DPO training.
python scripts/math/deepseek_math_sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/numina/830k-split-[23]-of-10/cot.de_con.[23]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/numina/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.json \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --filter_all_same --sample_over_p 8

# Vanilla outcome DPO

python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/numina/830k-split-[23]-of-10/cot.de_con.[23]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/numina/cot.de_con.n16.tem1.0.p1.0.prefer_pair.by_sc_p0.5.json \
    --top_p 0.5
#Filtered 95299 samples.
#Total number of items: 165284
#Acc: 0.8421661784668143
#Pass at k: 0.42342271484233196
#No positive solutions: 0 / 165284
#No negative solutions: 22851 / 165284
#Num pairs: 1994151

# ====================================== Split 45; Start from split 01 -> 23 ======================================

python scripts/math/deepseek_math_sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter2.split01-23.H100.dp8.v1.4.s42/checkpoint-100/numina/830k-split-[45]-of-10/cot.de_con.[45]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter2.split01-23.H100.dp8.v1.4.s42/checkpoint-100/numina/cot.de_con.n16.tem1.0.p1.0.split45.upper0.8.r0.3.sample8.filter_same.json \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --filter_all_same --sample_over_p 8


python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter2.split01-23.H100.dp8.v1.4.s42/checkpoint-100/numina/830k-split-[45]-of-10/cot.de_con.[45]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter2.split01-23.H100.dp8.v1.4.s42/checkpoint-100/numina/cot.de_con.n16.tem1.0.p1.0.split45.prefer_pair.by_sc_p0.5.json \
    --top_p 0.5
#Filtered 76704 samples.
#Total number of items: 165284
#Acc: 0.8534996613230977
#Pass at k: 0.5359260424481499
#No positive solutions: 0 / 165284
#No negative solutions: 31129 / 165284
#Num pairs: 2403757


# ====================================== Split 67; Start from split 01 -> 23 -> 45 ======================================

python scripts/math/deepseek_math_sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/830k-split-[67]-of-10/cot.de_con.[67]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/cot.de_con.n16.tem1.0.p1.0.split67.upper0.8.r0.3.sample8.filter_same.json \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --filter_all_same --sample_over_p 8

#Number of prefixes: 1997762
#249773

python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/830k-split-[67]-of-10/cot.de_con.[67]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/cot.de_con.n16.tem1.0.p1.0.split45.prefer_pair.by_sc_p0.0.json \
    --top_p 0.0
#Filtered 0 samples.
#Total number of items: 165284
#Acc: 0.7186297524261271
#Pass at k: 1.0
#No positive solutions: 0 / 165284
#No negative solutions: 32968 / 165284
#Num pairs: 6000694

# Split 89
python scripts/math/deepseek_math_sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/830k-split-[89]-of-10/cot.de_con.[89]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/cot.de_con.n16.tem1.0.p1.0.split89.upper0.8.r0.3.sample8.filter_same.json \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --filter_all_same --sample_over_p 8

python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/830k-split-[89]-of-10/cot.de_con.[89]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/cot.de_con.n16.tem1.0.p1.0.split89.prefer_pair.by_sc_p0.0.json \
    --top_p 0.0
#Filtered 0 samples.
#Total number of items: 165282
#Acc: 0.7201631151607556
#Pass at k: 1.0
#No positive solutions: 0 / 165282
#No negative solutions: 33060 / 165282
#Num pairs: 5986473
python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/830k-split-[89]-of-10/cot.de_con.[89]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/cot.de_con.n16.tem1.0.p1.0.split89.prefer_pair.by_sc_p0.2.json \
    --top_p 0.2
#Filtered 26351 samples.
#Total number of items: 165282
#Acc: 0.7397916951580281
#Pass at k: 0.8405694509988989
#No positive solutions: 0 / 165282
#No negative solutions: 33058 / 165282
#Num pairs: 5182543


python scripts/math/deepseek_math_sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.2.iter4.split01-23-45-67.H100.dp8.v1.5.s42/checkpoint-500/numina/830k-split-[89]-of-10/cot.de_con.[89]-of-10.n8.tem1.2.p1.0.*-of-64.s[01].json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.2.iter4.split01-23-45-67.H100.dp8.v1.5.s42/checkpoint-500/numina/cot.de_con.n16.tem1.2.p1.0.split89.upper0.8.r0.3.sample32.filter_same.json \
    --upper_step_ratio 0.8 --sample_ratio 0.3 --filter_all_same --sample_over_p 32
#Number of prefixes: 8049954
#254713



