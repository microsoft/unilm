export OUTPUT_PREFIX_PATH=/mnt/fangkai_blob/reward_modeling/

p=$1

#python scripts/math_scale/construct_process_rm_sample_gd.py \
#    --input_file "$OUTPUT_PREFIX_PATH/experiments/mathstral.mathscale4o.process-dpo-sc.iter0.A100.dp8.v2.1.s42/checkpoint-300/mathscale4o/split-1024/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
#    --output_file $OUTPUT_PREFIX_PATH/experiments/mathstral.mathscale4o.process-dpo-sc.iter0.A100.dp8.v2.1.s42/checkpoint-300/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.process_rm.gd.azure.json \
#    --num_workers 128

#= ================== mscale-300k

#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo-sc.iter0.A100.dp8.v2.1.s42/checkpoint-300/mathscale4o/mscale-v0.1-300k/split-1024/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo-sc.iter0.A100.dp8.v2.1.s42/checkpoint-300/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo-sc.iter0.A100.dp8.v2.1.s42/checkpoint-300/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" --response_id_field id --num_workers 128 --top_p $p
#

#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/mscale-v0.1-300k/split-1024/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" --response_id_field id --num_workers 128 --top_p $p

# ===================== 4o again

#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "$OUTPUT_PREFIX_PATH/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.iter0.H100.dp8.v1.3.s42/checkpoint-1200/mathscale4o/split-1024/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
#    --output_file $OUTPUT_PREFIX_PATH/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.iter0.H100.dp8.v1.3.s42/checkpoint-1200/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "$OUTPUT_PREFIX_PATH/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.iter0.H100.dp8.v1.3.s42/checkpoint-1200/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-64.s42.json" \
#    --response_id_field id --num_workers 128 --top_p $p
python scripts/math_scale/construct_process_rm_sample_gd.py \
    --input_file "$OUTPUT_PREFIX_PATH/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.iter0.H100.dp8.v1.3.s42/checkpoint-1200/mathscale4o/split-1024/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
    --output_file $OUTPUT_PREFIX_PATH/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.iter0.H100.dp8.v1.3.s42/checkpoint-1200/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.process_rm.gd.azure.json \
    --num_workers 128


