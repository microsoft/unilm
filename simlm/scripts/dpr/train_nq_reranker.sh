#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../../ && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/nq_rerank_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/dpr/"
fi

mkdir -p "${OUTPUT_DIR}"

# For electra-large, learning rate > 1e-5 will lead to instability empirically
PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
#python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} src/train_cross_encoder.py \
deepspeed src/train_cross_encoder.py --deepspeed ds_config.json \
    --model_name_or_path google/electra-base-discriminator \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --fp16 \
    --seed 987 \
    --train_file "${DATA_DIR}/nq_train.jsonl,${DATA_DIR}/nq_hard_train.jsonl" \
    --validation_file "${DATA_DIR}/nq_dev.jsonl" \
    --rerank_max_length 224 \
    --rerank_use_rdrop True \
    --use_first_positive True \
    --train_n_passages 32 \
    --rerank_forward_factor 2 \
    --dataloader_num_workers 1 \
    --learning_rate 3e-5 \
    --warmup_steps 1000 \
    --max_steps 20000 \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 5 \
    --save_strategy steps \
    --save_steps 2000 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --load_best_model_at_end \
    --metric_for_best_model acc \
    --greater_is_better True \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"
