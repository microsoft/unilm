cp=$1
cn=$2
exp_name=$3
n_gpus=$4
split_id_offset=$5

for ((i=6;i<=$#;i++)); do
  step=${!i}
  echo $step
  for j in {0..$(($n_gpus-1))}; do
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$j python vllm_inference.py eval_sub_path=checkpoint-$step output_path_prefix=/mnt/fangkai_blob/reward_modeling/ fp16_bfloat16=False split_id=$(($j+$split_id_offset)) exp_name=$exp_name -cp $cp -cn $cn &
  done
  wait
  echo "Step-$step Finished"
done

echo "Finished"
