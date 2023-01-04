#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../../ && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/nq_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/dpr/"
fi

mkdir -p "${OUTPUT_DIR}"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
# python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} src/train_biencoder.py \
deepspeed src/train_biencoder.py --deepspeed dpr_ds_config.json \
    --model_name_or_path intfloat/simlm-base-wiki100w \
    --task_type qa \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --l2_normalize True \
    --t 0.02 \
    --t_warmup True \
    --seed 5678 \
    --do_train \
    --fp16 \
    --train_file "${DATA_DIR}/nq_train.jsonl,${DATA_DIR}/nq_hard_train.jsonl" \
    --validation_file "${DATA_DIR}/nq_dev.jsonl" \
    --q_max_len 32 \
    --p_max_len 192 \
    --train_n_passages 16 \
    --use_first_positive True \
    --dataloader_num_workers 1 \
    --learning_rate 1e-5 \
    --use_scaled_loss True \
    --loss_scale 1 \
    --max_steps 20000 \
    --warmup_steps 1000 \
    --share_encoder True \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 3 \
    --save_strategy steps \
    --save_steps 2000 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"
