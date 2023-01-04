#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

mkdir -p data/

MSMARCO_BM25="msmarco_bm25_official.zip"
if [ ! -e data/$MSMARCO_BM25 ]; then
  wget -O data/${MSMARCO_BM25} https://huggingface.co/datasets/intfloat/simlm-msmarco/resolve/main/${MSMARCO_BM25}
  unzip data/${MSMARCO_BM25} -d data/
fi

MSMARCO_DISTILL="msmarco_distillation.zip"
if [ ! -e data/$MSMARCO_DISTILL ]; then
  wget -O data/${MSMARCO_DISTILL} https://huggingface.co/datasets/intfloat/simlm-msmarco/resolve/main/${MSMARCO_DISTILL}
  unzip data/${MSMARCO_DISTILL} -d data/
fi

MSMARCO_RERANK="msmarco_reranker.zip"
if [ ! -e data/$MSMARCO_RERANK ]; then
  wget -O data/${MSMARCO_RERANK} https://huggingface.co/datasets/intfloat/simlm-msmarco/resolve/main/${MSMARCO_RERANK}
  unzip data/${MSMARCO_RERANK} -d data/
fi

echo "data downloaded"
