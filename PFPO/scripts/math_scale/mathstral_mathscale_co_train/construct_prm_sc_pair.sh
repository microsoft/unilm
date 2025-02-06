export OUTPUT_PREFIX_PATH=/mnt/fangkai_blob/reward_modeling/

p=$1
half_id=$2

echo "Constructing process_rm sample for split $half_id"
echo "p" $p

python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-${half_id}.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-${half_id}-of-2/split-512/train.500k-${half_id}-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-${half_id}.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-${half_id}-of-2/train.500k-${half_id}-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-${half_id}.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-${half_id}-of-2/train.500k-${half_id}-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.*-of-256.s42.json" --response_id_field id --num_workers 128 --top_p $p
