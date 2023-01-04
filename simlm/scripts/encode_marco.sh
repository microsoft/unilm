#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=""
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/msmarco_bm25_official/"
fi

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=src/ python -u src/inference/encode_main.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_encode \
    --fp16 \
    --encode_in_path "${DATA_DIR}/passages.jsonl.gz" \
    --encode_save_dir "${OUTPUT_DIR}" \
    --encode_batch_size 512 \
    --p_max_len 144 \
    --add_pooler False \
    --dataloader_num_workers 1 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@"
