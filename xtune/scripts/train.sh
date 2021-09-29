#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
SETTING=${1:-cross-lingual-transfer}
TASK=${2:-xnli}
MODEL=${3:-"xlm-roberta-base"}
STAGE=${4:-1}
GPU=${5:-0}
DATA_DIR=${6:-"$REPO/download/"}
OUT_DIR=${7:-"$REPO/outputs/"}
SEED=${8:-1}

echo "Fine-tuning $MODEL on $TASK using GPU $GPU in STAGE $STAGE with SETTING $SETTING"
echo "Load data from $DATA_DIR, and save models to $OUT_DIR"

if [ $TASK == "udpos" ]; then
  bash $REPO/scripts/preprocess_udpos.sh $MODEL $DATA_DIR
elif [ $TASK == "panx" ]; then
  bash $REPO/scripts/preprocess_panx.sh $MODEL $DATA_DIR
fi

bash $REPO/scripts/$SETTING/train_${TASK}.sh $MODEL $STAGE $GPU $DATA_DIR $OUT_DIR $SEED