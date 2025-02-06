python scripts/math_scale/construct_prefer_pair.py \
  --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.*-of-512.s42.json" \
  --output_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.prefer_pair.4o.json"
#Total number of items: 298275
#Acc: 0.7078333752409689                                                                                                                                                                                                                                  Pass at k: 0.8356214902355209
#No positive solutions: 49030 / 298275                                                                                                                                                                                                                    No negative solutions: 148203 / 298275
#Num pairs: 1650384

python scripts/math_scale/construct_process_rm_sample_gd.py \
  --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/split-1024/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-1024.json" \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/mscale-v0.1-300k/mscale.v0.1.300k.v1.0.n10.tem1.0.p1.0.upper0.7.r0.3.sample10.filter_same.process_rm.4o.json \
  --num_workers 24
#Counter({0: 648447, 3: 465424, 2: 184121, 1: 178847, 6: 3645, 4: 1485})
#Missing 308 items in the response data.
#Counter({0: 648147, 3: 465044, 2: 183998, 1: 178757, 6: 3642, 4: 1484})
