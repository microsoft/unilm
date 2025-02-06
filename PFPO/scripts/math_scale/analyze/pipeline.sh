#exp_dir=llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/
#exp_dir=llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/
exp_dir=llama3.1.8b.numina.process-dpo-sc-p0.5.iter2.split01-23.H100.dp8.v1.4.s42/checkpoint-100/


#python scripts/math_scale/analyze/get_output_frequency.py \
#  --input_file "../msranlpintern/reward_modeling/experiments/${exp_dir}/math/math.test.v1.1.0shot.n1.tem1.0.p1.0.0-of-1.s*[0-9].json" \
#  --output_file ../msranlpintern/reward_modeling/experiments/${exp_dir}/math/math.test.v1.1.0shot.n64.tem1.0.p1.0.analyze.output_freq.json \
#  --n 128 --ps "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"


#python scripts/math_scale/analyze/freq2image.py \
#  --input_file ../msranlpintern/reward_modeling/experiments/${exp_dir}/math/math.test.v1.1.0shot.n64.tem1.0.p1.0.analyze.output_freq.json \
#  --output_file ./math.test.v1.1.0shot.n64.tem1.0.p1.0.analyze.output_freq.iter0.png

#exp_dir=llama3.1.8b.numina.process-dpo-sc-p0.5.iter1.split01.H100.dp8.v1.0.s42/checkpoint-200/
#python scripts/math_scale/analyze/freq2image.py \
#  --input_file ../msranlpintern/reward_modeling/experiments/${exp_dir}/math/math.test.v1.1.0shot.n64.tem1.0.p1.0.analyze.output_freq.json \
#  --output_file ./math.test.v1.1.0shot.n64.tem1.0.p1.0.analyze.output_freq.iter1.png
#
#exp_dir=llama3.1.8b.numina.process-dpo-sc-p0.5.iter2.split01-23.H100.dp8.v1.4.s42/checkpoint-100/
python scripts/math_scale/analyze/freq2image.py \
  --input_file ../msranlpintern/reward_modeling/experiments/${exp_dir}/math/math.test.v1.1.0shot.n64.tem1.0.p1.0.analyze.output_freq.json \
  --output_file ./math.test.v1.1.0shot.n64.tem1.0.p1.0.analyze.output_freq.iter2.png
