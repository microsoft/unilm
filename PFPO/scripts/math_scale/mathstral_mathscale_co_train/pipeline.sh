export OUTPUT_PREFIX_PATH=../msranlpintern/reward_modeling/

python scripts/math/deepseek_math_sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-0.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-0-of-2/train.500k-0-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.*-of-256.s42.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-0.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-0-of-2/train.500k-0-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 10

python scripts/math/deepseek_math_sample_steps.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-1.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-1-of-2/train.500k-1-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.*-of-256.s42.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-1.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-1-of-2/train.500k-1-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 10

python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-${half_id}.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-${half_id}-of-2/split-512/train.500k-${half_id}-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-${half_id}.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-${half_id}-of-2/train.500k-${half_id}-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-${half_id}.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-${half_id}-of-2/train.500k-${half_id}-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.*-of-256.s42.json" --response_id_field id --num_workers 128 --top_p $p

python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-${half_id}.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-${half_id}-of-2/split-512/train.500k-${half_id}-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-${half_id}.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-${half_id}-of-2/train.500k-${half_id}-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.process_rm.sc-p$p.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.co-sft.half-${half_id}.V100.tp2dp64.v1.0.s42/checkpoint-400/mathscale4o/500k-half-${half_id}-of-2/train.500k-${half_id}-of-2.de_con.boxed.v1.0.n10.tem1.0.p0.9.*-of-256.s42.json" --response_id_field id --num_workers 128 --top_p $p

# half=1 p=0.0
#Counter({3: 555266, 0: 429788, 2: 288700, 1: 262752})
#Missing 9618 items in the response data.
#Counter({3: 552136, 0: 426893, 2: 286803, 1: 261009})
# half=1 p=0.5
#Counter({3: 511946, 2: 218546, 1: 140566, 0: 122720})
#Missing 552346 items in the response data.
#Counter({3: 509128, 2: 217127, 1: 139676, 0: 121953})
# half=0 p=0.5
#Counter({3: 538985, 2: 229498, 1: 146411, 0: 128951})
#Missing 579971 items in the response data.
#Counter({3: 521894, 2: 223010, 1: 142343, 0: 125221})
# half=0 p=0.0
# Counter({3: 584337, 0: 451324, 2: 303606, 1: 274166})
# Missing 10383 items in the response data.
# Counter({3: 566162, 0: 440860, 2: 295316, 1: 267302})