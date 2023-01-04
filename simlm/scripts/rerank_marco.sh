#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=""
RERANK_IN_PATH=""
SPLIT="dev"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    RERANK_IN_PATH=$1
    shift
fi

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SPLIT=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/msmarco_bm25_official/"
fi

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=src/ python -u src/inference/rerank_main.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_rerank \
    --fp16 \
    --rerank_in_path "${RERANK_IN_PATH}" \
    --rerank_out_path "${OUTPUT_DIR}/rerank.${SPLIT}.msmarco.txt" \
    --rerank_batch_size 256 \
    --rerank_max_length 192 \
    --rerank_split "${SPLIT}" \
    --rerank_depth 200 \
    --dataloader_num_workers 1 \
    --output_dir "/tmp/" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@"
