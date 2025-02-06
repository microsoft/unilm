path=$1

shift 1 # Shift the first 5 arguments, so $1 now refers to the 6th argument

for step in "$@"; do
  echo "Merging predictions for step $step"
  python scripts/math_scale/merge_dp_predictions.py --input_file "$path/mwpbench/checkpoint-$step/train_wo_math.2k.v1.3.0shot.n1.tem0.0.p1.0.?-of-8.json" \
    --output_file "$path/mwpbench/checkpoint-$step/train_wo_math.2k.v1.3.0shot.n1.tem0.0.p1.0.json"
  python scripts/math_scale/merge_dp_predictions.py --input_file "$path/mwpbench/checkpoint-$step/full_test.v1.3.0shot.n1.tem0.0.p1.0.?-of-8.json" \
    --output_file "$path/mwpbench/checkpoint-$step/full_test.v1.3.0shot.n1.tem0.0.p1.0.json"
  echo "Done merging predictions for step $step"
  echo
done