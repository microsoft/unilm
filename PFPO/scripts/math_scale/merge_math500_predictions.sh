path=$1
prefix=$2
version=$3

split_size=1

shift 3 # Shift the first 5 arguments, so $1 now refers to the 6th argument

for step in "$@"; do
  echo "Merging predictions for step $step"
#  python scripts/math_scale/merge_dp_predictions.py --input_file "$path/math500/checkpoint-$step/$prefix.test.v$version.0shot.n1.tem0.0.p1.0.?-of-${split_size}.json" \
#    --output_file "$path/math500/checkpoint-$step/$prefix.test.v$version.0shot.n1.tem0.0.p1.0.json"
  cat "$path/math500/checkpoint-$step/$prefix.test.v$version.0shot.n1.tem0.0.p1.0.0-of-${split_size}.metrics.json"
  echo "Done merging predictions for step $step"
  echo
done