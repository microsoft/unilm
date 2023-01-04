#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/rlm_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/msmarco_bm25_official/"
fi

mkdir -p "${OUTPUT_DIR}"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} src/train_rlm.py \
deepspeed src/train_rlm.py --deepspeed ds_config.json \
    --model_name_or_path bert-base-uncased \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --seed 45678 \
    --do_train \
    --do_eval \
    --fp16 \
    --train_file "${DATA_DIR}/passages.jsonl.gz" \
    --rlm_max_length 144 \
    --rlm_encoder_mask_prob 0.3 \
    --rlm_decoder_mask_prob 0.5 \
    --rlm_generator_model_name google/electra-base-generator \
    --rlm_freeze_generator True \
    --rlm_generator_mlm_weight 0.2 \
    --all_use_mask_token True \
    --dataloader_num_workers 1 \
    --max_steps 80000 \
    --learning_rate 3e-4 \
    --warmup_steps 4000 \
    --weight_decay 0.0 \
    --remove_unused_columns False \
    --logging_steps 50 \
    --report_to none \
    --output_dir "${OUTPUT_DIR}" \
    --save_total_limit 20 \
    --save_strategy steps \
    --save_steps 10000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --data_dir "${DATA_DIR}" \
    --overwrite_output_dir \
    --disable_tqdm True "$@"
