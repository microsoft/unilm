#!/bin/bash

set -ex

gpu_rank=$1
quantize_size=$2
model_path=$3
eval_json=$4
ann_json=$5

dir_path=`dirname $model_path`
file_name=`basename $model_path`

eval_name=$(echo "$eval_json" | sed -E 's/.*finetune_(.*)\.json.*/\1/' | sed 's/_/./') # .refcoco.testA | .refcocog.test ....
template_post_name=$(echo "$eval_json" | sed 's/.*\.json//' )
eval_out_post_name=".eval.${eval_name}${template_post_name}"
result_post_name="${eval_out_post_name}.result"

master_port=$((RANDOM%1000+20000))

cat $eval_json | CUDA_VISIBLE_DEVICES=$gpu_rank python -m torch.distributed.launch --nproc_per_node=1 --master_port=$master_port evaluation/caption_obj_few_shot.py None \
    --task generation_obj \
    --path $model_path \
    --model-overrides "{'visual_pretrained': '',
            'dict_path':'data/dict.txt'}" \
    --dict-path 'data/dict.txt' \
    --required-batch-size-multiple 1 \
    --remove-bpe=sentencepiece \
    --max-len-b 500 \
    --add-bos-token \
    --beam 1 \
    --buffer-size 10 \
    --image-feature-length 64 \
    --locate-special-token 1 \
    --batch-size 1 \
    --nbest 1 \
    --no-repeat-ngram-size 3 \
    --location-bin-size $quantize_size > $dir_path/${file_name}$eval_out_post_name

python evaluation/refcoco/refexp_evaluate.py $dir_path/${file_name}$eval_out_post_name ${ann_json} --quantized_size $quantize_size > $dir_path/${file_name}$result_post_name
tail -n 6 $dir_path/${file_name}$result_post_name