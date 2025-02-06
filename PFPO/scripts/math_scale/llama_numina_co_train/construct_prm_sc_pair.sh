export OUTPUT_PREFIX_PATH=/mnt/fangkai_blob/reward_modeling/

p=$1

echo "Constructing process_rm sample for split 0"
echo "p" $p

#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split01.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-512.s0.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split01.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/830k-split-[01]-of-10/cot.de_con.[01]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p
#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-512.s0.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/830k-split-[23]-of-10/cot.de_con.[23]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p

# =================================================================================================

# Use the co-part model's responses as sc labels.
#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split01.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-512.s0.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split23.A100.dp16.v1.0.s42/cot.de_con.n16.tem1.0.p1.0.split01.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split23.A100.dp16.v1.0.s42/checkpoint-100/numina/830k-split-[01]-of-10/cot.de_con.[01]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p

#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-512.s0.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/numina/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/numina/830k-split-[23]-of-10/cot.de_con.[23]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p


# Iterative DPO: split-01 -> split-23
#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/numina/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-1024.s0.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/numina/cot.de_con.n16.tem1.0.p1.0.split23.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/numina/830k-split-[23]-of-10/cot.de_con.[23]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p

# Iterative DPO: split-01 -> split-23 -> split-45
#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter2.split01-23.H100.dp8.v1.4.s42/checkpoint-100/numina/cot.de_con.n16.tem1.0.p1.0.split45.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-1024.s0.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter2.split01-23.H100.dp8.v1.4.s42/checkpoint-100/numina/cot.de_con.n16.tem1.0.p1.0.split45.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter2.split01-23.H100.dp8.v1.4.s42/checkpoint-100/numina/830k-split-[45]-of-10/cot.de_con.[45]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p

# Iterative DPO: split-01 -> split-23 -> split-45 -> split-67
#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/cot.de_con.n16.tem1.0.p1.0.split67.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-1024.s0.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/cot.de_con.n16.tem1.0.p1.0.split67.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/830k-split-[67]-of-10/cot.de_con.[67]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p

# Iterative DPO: split-01 -> split-23 -> split-45 -> split-89
#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/cot.de_con.n16.tem1.0.p1.0.split89.upper0.8.r0.3.sample8.filter_same.n3.tem1.0.p1.0.*-of-1024.s0.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/cot.de_con.n16.tem1.0.p1.0.split89.upper0.8.r0.3.sample8.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.5.iter3.split01-23-45.H100.dp8.v1.4.s42/checkpoint-400/numina/830k-split-[89]-of-10/cot.de_con.[89]-of-10.n8.tem1.0.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p

# Iterative DPO: split-01 -> split-23 -> split-45 -> split-67 -> split-89
python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.2.iter4.split01-23-45-67.H100.dp8.v1.5.s42/checkpoint-500/numina/cot.de_con.n16.tem1.2.p1.0.split89.upper0.8.r0.3.sample32.filter_same.n3.tem1.0.p1.0.*-of-1024.s0.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.2.iter4.split01-23-45-67.H100.dp8.v1.5.s42/checkpoint-500/numina/cot.de_con.n16.tem1.2.p1.0.split89.upper0.8.r0.3.sample32.filter_same.process_rm.sc-p$p.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/llama3.1.8b.numina.process-dpo-sc-p0.2.iter4.split01-23-45-67.H100.dp8.v1.5.s42/checkpoint-500/numina/830k-split-[89]-of-10/cot.de_con.[89]-of-10.n8.tem1.2.p1.0.*-of-64.s[01].json" --response_id_field id --num_workers 128 --top_p $p
