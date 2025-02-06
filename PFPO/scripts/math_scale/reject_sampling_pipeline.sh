# Iter - 1: PRM reranking
export OUTPUT_PATH_PREFIX=../msranlpintern/reward_modeling

for global_split_id in {0..19}; do
  python scripts/math_scale/rerank_w_prm_math_scale_save.py \
    --response_file "${OUTPUT_PATH_PREFIX}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-${global_split_id}-of-20/train.500k.de_con.boxed.v1.0.${global_split_id}-of-20.0shot.n90.tem1.0.p0.9.json" \
    --reward_file "${OUTPUT_PATH_PREFIX}/experiments/mathstral.mathscale4o.process-rm-sc.iter1.h100.dp8.v1.0.s42/mathstral.mathscale4o.pdpo.iter0.v2.2.s42.ckpt-600.mathscale4o.500k.global-${global_split_id}-of-20-local-*-of-30/test-checkpoint-2000/eval_predictions.json" \
    --reduction "min" --num_workers 64 --top_k 4 --sc_type "bon" --output_file ${OUTPUT_PATH_PREFIX}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n90.tem1.0.p0.9.sc-iter1-prm-top4.min.bon.${global_split_id}-of-20. json
done

# Concat dataset
python scripts/math_scale/concat_data.py \
  --input_file "${OUTPUT_PATH_PREFIX}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n90.tem1.0.p0.9.sc-iter1-prm-top4.min.bon.w_incorrect.glo-*-of-20.json" \
  --output_file ${OUTPUT_PATH_PREFIX}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n90.tem1.0.p0.9.sc-iter1-prm-top4.min.bon.w_incorrect.json


python scripts/math_scale/concat_data.py \
  --input_file "${OUTPUT_PATH_PREFIX}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n90.tem1.0.p0.9.sc-iter1-prm-top4.min.bon.glo-*-of-20.json" \
  --output_file ${OUTPUT_PATH_PREFIX}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n90.tem1.0.p0.9.sc-iter1-prm-top4.min.bon.json


# Iter - 1: DPO - NuminaMath (new questions)
python scripts/math_scale/construct_prefer_pair_sc.py \
  --input_file "${OUTPUT_PATH_PREFIX}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.*-of-8.s0.json" \
  --output_file $OUTPUT_PATH_PREFIX/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.s0.prefer_pair.by_sc.json

python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/completion-split-128/cot.de_con.n8.tem1.0.p1.0.s0.upper0.7.r0.3.sample32.filter_same.{split_id}-of-4.n3.tem1.0.p1.0.*-of-128.s0.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/cot.de_con.n8.tem1.0.p1.0.s0.upper0.7.r0.3.sample32.filter_same.process_rm.iter0-pdpo-sc-p0.5.v0.0.{split_id}-of-4.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-*-of-10/cot.de_con.*-of-10.n8.tem1.0.p1.0.*-of-*.s*.json" --response_id_field id --num_workers 128 --top_p 0.5



