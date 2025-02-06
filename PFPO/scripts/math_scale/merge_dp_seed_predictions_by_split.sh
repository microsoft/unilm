for i in {0..19}; do
     python scripts/math_scale/merge_dp_seed_predictions.py \
     --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-${i}-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.[0-8]-of-8.s[0-8].json" \
     --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-${i}-of-20/train.500k.de_con.boxed.v1.0.${i}-of-20.0shot.n90.tem1.0.p0.9.json
done