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
DIR=$REPO/download/
mkdir -p $DIR

# download xlm-roberta-base
function download_xlm-roberta-base {
    mkdir -p $DIR/xlm-roberta-base/
    cd $DIR/xlm-roberta-base/
    wget https://huggingface.co/xlm-roberta-base/resolve/main/pytorch_model.bin -q --show-progress
    wget https://huggingface.co/xlm-roberta-base/resolve/main/config.json -q --show-progress
    wget https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model -q --show-progress
    wget https://huggingface.co/xlm-roberta-base/resolve/main/tokenizer.json -q --show-progress
    echo "Successfully downloaded xlm-roberta-base at $DIR/xlm-roberta-base" >> $DIR/download_model.log
}

# download xlm-roberta-large
function download_xlm-roberta-large {
    mkdir -p $DIR/xlm-roberta-large/
    cd $DIR/xlm-roberta-large/
    wget https://huggingface.co/xlm-roberta-large/resolve/main/pytorch_model.bin -q --show-progress
    wget https://huggingface.co/xlm-roberta-large/resolve/main/config.json -q --show-progress
    wget https://huggingface.co/xlm-roberta-large/resolve/main/sentencepiece.bpe.model -q --show-progress
    wget https://huggingface.co/xlm-roberta-large/resolve/main/tokenizer.json -q --show-progress
    echo "Successfully downloaded xlm-roberta-large at $DIR/xlm-roberta-large" >> $DIR/download_model.log
}

download_xlm-roberta-base
download_xlm-roberta-large
