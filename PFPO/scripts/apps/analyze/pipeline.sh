exp_dir="../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.r2c.sft_ps_test_case.iter2.dpo.H100.dp8.v1.0.s42/apps/checkpoint-2400"

#python scripts/apps/analyze/get_output_frequency.py \
#  --completion_file "${exp_dir}/test.0shot.tem1.0.n8.*-of-32.v2.0.json" \
#  --exec_file "${exp_dir}/test.0shot.tem1.0.n8.*-of-32.v2.0.run_outputs.json" \
#  --output_file ${exp_dir}/test.0shot.tem1.0.n8.v2.0.run_outputs.analyze_freq.json --num_workers 16


#python scripts/apps/analyze/freq2image.py \
#  --freq_file ${exp_dir}/test.0shot.tem1.0.n8.v2.0.run_outputs.analyze_freq.json \
#  --output_file ./apps.test.0shot.tem1.0.n8.v2.0.analyze_freq.png


#python scripts/apps/analyze/get_output_frequency.py \
#  --completion_file "${exp_dir}/test.0shot.tem1.0.n8.*-of-32.v2.0.s?.json" \
#  --exec_file "${exp_dir}/test.0shot.tem1.0.n8.*-of-32.v2.0.s?.run_outputs.json" \
#  --output_file ${exp_dir}/test.0shot.tem1.0.n64.v2.0.run_outputs.analyze_freq.json --num_workers 16

#Self-consistency: 1067/5000 = 0.2134
#Program self-consistency: 1053/5000 = 0.2106
#First res: 883/5000 = 0.1766
#Programs: 320000/5000 = 64.0

python scripts/apps/analyze/freq2image.py \
  --freq_file ${exp_dir}/test.0shot.tem1.0.n64.v2.0.run_outputs.analyze_freq.json  \
  --output_file ./apps.test.0shot.tem1.0.n64.v2.0.analyze_freq.png