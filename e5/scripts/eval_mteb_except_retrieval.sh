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
  OUTPUT_DIR="tmp-outputs/"
fi

python -u mteb_except_retrieval_eval.py \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --task-types "STS" "Summarization" "PairClassification" "Classification" "Reranking" "Clustering" "BitextMining" \
    --output-dir "${OUTPUT_DIR}" "$@"

echo "done"
