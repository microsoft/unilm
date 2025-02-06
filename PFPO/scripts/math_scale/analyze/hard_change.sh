python scripts/math_scale/analyze/extract_hard_questions.py \
  --input_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.json \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.k16-fail.json \
  --maj_k 16

python scripts/math_scale/analyze/compute_acc_by_id.py \
  --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n1.tem1.0.p1.0.0-of-1.s*.json" \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.sft-k16-fail.json \
  --id_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.k16-fail.json

python scripts/math_scale/qwen25math_style_eval_math.py \
  --input_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.json \
  --num_workers 1
#{
#  "acc": 0.6292,
#  "pass@k": 0.6292,
#  "maj@k": 0.6292,
#  "correct": 3146,
#  "total": 5000
#}
# Compared with original: 61.4
python scripts/math_scale/analyze/compute_acc_by_id.py \
  --input_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.sympy_eval.json \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.sympy_eval.sft-k16-fail.json \
  --id_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.k16-fail.json

python scripts/math_scale/analyze/compute_acc_by_id.py \
  --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/math/math.test.v1.1.0shot.n1.tem1.0.p1.0.0-of-1.s*.json" \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.sft-k16-fail.json \
  --id_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.k16-fail.json

python scripts/math_scale/qwen25math_style_eval_math.py \
  --input_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.json \
  --num_workers 1
#{
#  "acc": 0.6702,
#  "pass@k": 0.6702,
#  "maj@k": 0.6702,
#  "correct": 3351,
#  "total": 5000
#}
python scripts/math_scale/analyze/compute_acc_by_id.py \
  --input_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.sympy_eval.json \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/math.test.v1.1.0shot.n1.tem0.0.p1.0.sympy_eval.sft-k16-fail.json \
  --id_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.k16-fail.json


python scripts/math_scale/analyze/compute_acc_by_id.py \
  --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/math/math.test.v1.1.0shot.n1.tem1.0.p1.0.0-of-1.s*.json" \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.sft-k16-fail.json \
  --id_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.k16-fail.json
python scripts/math_scale/qwen25math_style_eval_math.py \
  --input_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.json \
  --num_workers 1
python scripts/math_scale/analyze/compute_acc_by_id.py \
  --input_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.sympy_eval.json \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42/checkpoint-500/math.test.v1.1.0shot.n1.tem0.0.p1.0.sympy_eval.sft-k16-fail.json \
  --id_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.k16-fail.json

# mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42
python scripts/math_scale/analyze/compute_acc_by_id.py \
  --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/math/math.test.v1.1.0shot.n1.tem1.0.p1.0.0-of-1.s*.json" \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.sft-k16-fail.json \
  --id_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.k16-fail.json
python scripts/math_scale/qwen25math_style_eval_math.py \
  --input_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.json \
  --num_workers 1
python scripts/math_scale/analyze/compute_acc_by_id.py \
  --input_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.sympy_eval.json \
  --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mscale-v0.1-300k.process-dpo-sc.p0.0.iter2.H100.dp16.v1.3.s42/checkpoint-500/math.test.v1.1.0shot.n1.tem0.0.p1.0.sympy_eval.sft-k16-fail.json \
  --id_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/math/math.test.v1.1.0shot.n128.tem1.0.p1.0.k16-fail.json
