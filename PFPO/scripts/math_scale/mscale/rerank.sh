export OUTPUT_PREFIX_PATH=/mnt/fangkai_blob/reward_modeling/

target_dir=$1

#python scripts/math_scale/rerank_w_prm_math.py \
#  --response_file "$OUTPUT_PREFIX_PATH/experiments/$target_dir/math/math.test.v1.1.0shot.n1.tem1.0.p1.0.0-of-1.s*[0-9].json" \
#  --reward_file "$OUTPUT_PREFIX_PATH/experiments/mathstral.mscale-v0.1-300k.process-rm-sc.p0.0.iter2.V100.dp256.v1.0.s42/$target_dir/-*/test-checkpoint-500/eval_predictions.json" \
#  --reduction min --num_workers 128 --re_index

python scripts/math_scale/rerank_w_prm_math.py \
  --response_file "$OUTPUT_PREFIX_PATH/experiments/$target_dir/math/math.test.v1.1.0shot.n1.tem1.0.p1.0.0-of-1.s*[0-9].json" \
  --reward_file "$OUTPUT_PREFIX_PATH/experiments/mathstral.mscale-v0.1-300k.process-rm-sc.p0.0.iter3.V100.dp256.v1.0.s42/$target_dir/-*/test-checkpoint-400/eval_predictions.json" \
  --reduction min --num_workers 128 --re_index


