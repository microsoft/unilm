#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../../ && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=""
RERANK_IN_PATH=""
SPLIT="nq_dev"
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
  DATA_DIR="${DIR}/data/dpr/"
fi

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=src/ python -u src/inference/rerank_main.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_rerank \
    --task_type qa \
    --fp16 \
    --rerank_in_path "${RERANK_IN_PATH}" \
    --rerank_out_path "${OUTPUT_DIR}/rerank.${SPLIT}.msmarco.txt" \
    --rerank_batch_size 128 \
    --rerank_max_length 224 \
    --rerank_split "${SPLIT}" \
    --rerank_depth 100 \
    --dataloader_num_workers 1 \
    --output_dir "/tmp/" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@"

python -u misc/dpr/format_and_evaluate.py \
  --data-dir "${DATA_DIR}" \
  --topk 1 5 20 100 \
  --topics "${DATA_DIR}/${SPLIT}_queries.tsv" \
  --input "${OUTPUT_DIR}/rerank.${SPLIT}.msmarco.txt" \
  --output "${OUTPUT_DIR}/${SPLIT}.dpr.json"
