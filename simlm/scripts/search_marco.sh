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

SPLIT="dev"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SPLIT=$1
    shift
fi

DEPTH=1000
# by default, search top-200 for train, top-1000 for dev
if [ "${SPLIT}" = "train" ]; then
  DEPTH=200
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/msmarco_bm25_official/"
fi

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=src/ python -u src/inference/search_main.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --search_split "${SPLIT}" \
    --search_batch_size 128 \
    --search_topk "${DEPTH}" \
    --search_out_dir "${OUTPUT_DIR}" \
    --encode_save_dir "${OUTPUT_DIR}" \
    --q_max_len 32 \
    --add_pooler False \
    --dataloader_num_workers 1 \
    --output_dir "/tmp/" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@"
