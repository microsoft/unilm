#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=""
SPLIT="dev"
DATA_DIR="${DIR}/data/msmarco_bm25_official/"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    DATA_DIR=$1
    shift
fi

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SPLIT=$1
    shift
fi

PYTHONPATH=src/ python -u src/inference/gen_teacher_scores.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_kd_gen_score \
    --fp16 \
    --data_dir "${DATA_DIR}" \
    --kd_gen_score_split "${SPLIT}" \
    --kd_gen_score_batch_size 256 \
    --kd_gen_score_n_neg 1000 \
    --rerank_max_length 192 \
    --dataloader_num_workers 1 \
    --output_dir "/tmp/" \
    --report_to none "$@"
