path=$1

shift 1 # Shift the first 5 arguments, so $1 now refers to the 6th argument

for step in "$@"; do
  echo "Merging predictions for step $step"
  cat "$path/human_eval/checkpoint-$step/test.0shot.tem0.0.n1.v1.0.metrics.json"
  cat "$path/mbpp_257/checkpoint-$step/test.0shot.tem0.0.n1.v1.0.metrics.json"
  cat "$path/mbpp_257/checkpoint-$step/test.3shot.tem0.0.n1.v2.0.metrics.json"
  echo "Done merging predictions for step $step"
  echo
done