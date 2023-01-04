#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/kd_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/msmarco_distillation/"
fi

mkdir -p "${OUTPUT_DIR}"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
# python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} src/train_biencoder.py \
deepspeed src/train_biencoder.py --deepspeed ds_config.json \
    --model_name_or_path intfloat/simlm-base-msmarco \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --kd_mask_hn False \
    --kd_cont_loss_weight 0.2 \
    --seed 123 \
    --do_train \
    --do_kd_biencoder \
    --t 0.02 \
    --fp16 \
    --train_file "${DATA_DIR}/kd_train.jsonl" \
    --validation_file "${DATA_DIR}/kd_dev.jsonl" \
    --q_max_len 32 \
    --p_max_len 144 \
    --train_n_passages 24 \
    --dataloader_num_workers 1 \
    --num_train_epochs 6 \
    --learning_rate 3e-5 \
    --warmup_steps 1000 \
    --share_encoder True \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model mrr \
    --greater_is_better True \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"
