
python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 10


# Vanilla outcome DPO

python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.prefer_pair.by_sc_p0.5.json \
    --top_p 0.5
#Filtered 63546 samples.
#Total number of items: 298275
#Acc: 0.9530181613690681
#Pass at k: 0.7869549911993965
#No positive solutions: 0 / 298275
#No negative solutions: 156892 / 298275
#Num pairs: 1233371

python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.prefer_pair.by_sc_p0.0.json \
    --top_p 0.0
#Filtered 0 samples.
#Total number of items: 298275
#Acc: 0.8771335177269298
#Pass at k: 1.0
#No positive solutions: 0 / 298275
#No negative solutions: 157701 / 298275
#Num pairs: 2538931


# Iter - 2
python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 10


python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.prefer_pair.by_sc_p0.5.json \
    --top_p 0.5

#Filtered 43373 samples.
#Total number of items: 298275
#Acc: 0.9614400828553719
#Pass at k: 0.8545872097896237
#No positive solutions: 0 / 298275
#No negative solutions: 192622 / 298275
#Num pairs: 986657

python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.prefer_pair.by_sc_p0.json \
    --top_p 0.0
#Filtered 0 samples.
#Total number of items: 298275
#Acc: 0.9059827340541446
#Pass at k: 1.0
#No positive solutions: 0 / 298275
#No negative solutions: 193482 / 298275
#Num pairs: 1897089


# Iter - 3
python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 10


python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.prefer_pair.by_sc_p0.0.json \
    --top_p 0.0