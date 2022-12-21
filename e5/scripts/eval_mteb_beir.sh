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

mkdir -p "${OUTPUT_DIR}"

python -u mteb_beir_eval.py \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --pool-type avg \
    --output-dir "${OUTPUT_DIR}" "$@"

echo "done"
