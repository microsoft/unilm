#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../../ && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=""
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

SPLIT="nq_dev"
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

PYTHONPATH=src/ python -u src/inference/search_main.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --task_type qa \
    --search_split "${SPLIT}" \
    --search_batch_size 128 \
    --search_topk 100 \
    --search_out_dir "${OUTPUT_DIR}" \
    --encode_save_dir "${OUTPUT_DIR}" \
    --q_max_len 32 \
    --l2_normalize True \
    --dataloader_num_workers 1 \
    --output_dir "/tmp/" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@"

python -u misc/dpr/format_and_evaluate.py \
  --data-dir "${DATA_DIR}" \
  --topk 1 5 20 100 \
  --topics "${DATA_DIR}/${SPLIT}_queries.tsv" \
  --input "${OUTPUT_DIR}/${SPLIT}.msmarco.txt" \
  --output "${OUTPUT_DIR}/${SPLIT}.dpr.json"
