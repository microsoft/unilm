export OUTPUT_PREFIX_PATH=/mnt/fangkai_blob/reward_modeling/

p=$1

echo "Constructing process_rm sample for split 0"
echo "p" $p

#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/split-1024/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" --response_id_field id --num_workers 128 --top_p $p#


#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/split-1024/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" --response_id_field id --num_workers 128 --top_p $p
#p=0.5
#Counter({3: 493281, 2: 165717, 1: 102586, 0: 86441})
#Missing 298249 items in the response data.
#Counter({3: 487316, 2: 163641, 1: 101298, 0: 85343})
#p=0
#Counter({3: 522458, 0: 241172, 2: 209424, 1: 172472})
#Missing 748 items in the response data.
#Counter({3: 516169, 0: 238263, 2: 206891, 1: 170353})

#python scripts/math_scale/construct_process_rm_sample_sc.py \
#    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/split-1024/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
#    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
#    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" --response_id_field id --num_workers 128 --top_p $p
#Counter({3: 470992, 0: 195492, 2: 178005, 1: 144163})
#Missing 391 items in the response data.
#Counter({3: 470540, 0: 195452, 2: 177903, 1: 144094})


python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/split-1024/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n8.tem1.0.p1.0.*-of-512.s*.json" --response_id_field id --num_workers 128 --top_p $p
