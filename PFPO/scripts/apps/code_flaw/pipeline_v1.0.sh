python scripts/apps/worsen_gpt4_combine.py \
  --worsen_file outputs/apps/critique/r2c.sft.train.0shot.tem1.0.n10.v1.1.worsen_4o_critic.s42.f100k.gpt4o.tem1.0.s42.n1.json_obj.jsonl \
  --completion_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.?-of-8.v1.1.json" \
  --completion_problem_id_field id \
  --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.v1.1.s43.gpt4o.worsen.f100k.json


python scripts/apps/rerank_code_rm.py \
  --completion_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_dpo.A100.dp8.v3.0.s42/apps/checkpoint-200/test.0shot.tem1.0.n20.v1.1.json \
  --reward_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.pair-rm.gpt4o-worsen.A100.w8.v1.0.s42/r2c.dpo.step-200.test.n20/test-checkpoint-50/eval_predictions_rank?.json"


python scripts/apps/rerank_code_rm.py \
  --completion_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_dpo.A100.dp8.v3.0.s42/apps/checkpoint-200/test.0shot.tem1.0.n20.v1.1.json \
  --reward_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.pair-rm.r2c-policy-data.A100.w8.v1.0.s42/r2c.dpo.step-200.test.n20/test-checkpoint-50/eval_predictions_rank?.json"