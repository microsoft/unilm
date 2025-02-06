path=$1
prefix=$2
version=$3

split_size=1

shift 3 # Shift the first 5 arguments, so $1 now refers to the 6th argument

for step in "$@"; do
  echo "Merging predictions for step $step"
  python scripts/math_scale/merge_dp_predictions.py --input_file "$path/mwpbench/checkpoint-$step/$prefix.test.v$version.0shot.n1.tem0.0.p1.0.?-of-${split_size}.json" \
    --output_file "$path/mwpbench/checkpoint-$step/$prefix.test.v$version.0shot.n1.tem0.0.p1.0.json"
#  python scripts/math_scale/merge_dp_predictions.py --input_file "$path/mwpbench/checkpoint-$step/full_test.v1.0.0shot.n1.tem0.0.p1.0.?-of-8.json" \
#    --output_file "$path/mwpbench/checkpoint-$step/full_test.v1.0.0shot.n1.tem0.0.p1.0.json"
#  python scripts/math_scale/merge_dp_predictions.py --input_file "$path/checkpoint-$step/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.?-of-8.json" \
#    --output_file "$path/checkpoint-$step/math/math.test.v1.1.0shot.n1.tem0.0.p1.0.json"
#  cat "$path/mwpbench/checkpoint-$step/fresh_gaokao_math_2023.v1.0.0shot.n1.tem0.0.p1.0.0-of-1.metrics.json"
  echo "Done merging predictions for step $step"
  echo
done