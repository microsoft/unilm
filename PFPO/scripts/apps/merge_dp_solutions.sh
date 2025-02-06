path=$1

shift 1 # Shift the first 5 arguments, so $1 now refers to the 6th argument

for step in "$@"; do
  echo "Merging predictions for checkpoint $step"
  python scripts/apps/merge_dp_predictions.py --input_file "$path/apps/checkpoint-$step/sub_dev.0shot.tem0.0.n1.?-of-8.v2.0.json" \
    --output_file "$path/apps/checkpoint-$step/sub_dev.0shot.tem0.0.n1.v2.0.json" --split train
  python scripts/apps/merge_dp_predictions.py --input_file "$path/apps/checkpoint-$step/test.0shot.tem0.0.n1.?-of-8.v2.0.json" \
    --output_file "$path/apps/checkpoint-$step/test.0shot.tem0.0.n1.v2.0.json" --split test
#  cat $path/apps/checkpoint-$step/sub_dev.0shot.tem0.0.n1.0-of-1.v2.0.metrics.json
#  cat $path/apps/checkpoint-$step/test.0shot.tem0.0.n1.0-of-1.v2.0.metrics.json
  echo "Done merging predictions for checkpoint $step"
  echo ""
done