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
MODEL=${1:-"xlm-roberta-base"}
STAGE=${2:-1}
GPU=${3:-0}
DATA_DIR=${4:-"$REPO/download/"}
OUT_DIR=${5:-"$REPO/outputs/"}
SEED=${6:-1}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='xnli'
TRANSLATION_PATH=$DATA_DIR/xtreme_translations/XNLI/
MODEL_PATH=$DATA_DIR/$MODEL
EPOCH=10
MAXL=256
LANGS="ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh"
EVALUATE_STEPS=5000
CSR=0.3
R1_LAMBDA=5.0
R2_LAMBDA=5.0
if [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=16
  GRAD_ACC=2
  LR=5e-6
else
  BATCH_SIZE=32
  GRAD_ACC=1
  LR=7e-6
fi

if [ $STAGE == 1 ]; then
  OUTPUT_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-CS-csr${CSR}-R1_LAMBDA${R1_LAMBDA}/"
  mkdir -p $OUTPUT_DIR
  python ./src/run_cls.py --model_type xlmr \
        --model_name_or_path $MODEL_PATH \
        --language $LANGS \
        --train_language en \
        --do_train \
        --data_dir $DATA_DIR/$TASK/ \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC \
        --per_gpu_eval_batch_size 64 \
        --learning_rate $LR \
        --num_train_epochs $EPOCH \
        --max_seq_length $MAXL \
        --output_dir $OUTPUT_DIR \
        --task_name $TASK \
        --save_steps -1 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --evaluate_steps $EVALUATE_STEPS \
        --logging_steps 50 \
        --logging_steps_in_sample -1 \
        --logging_each_epoch \
        --gpu_id 0 \
        --seed $SEED \
        --fp16 --fp16_opt_level O2 \
        --warmup_steps -1 \
        --enable_r1_loss \
        --r1_lambda $R1_LAMBDA \
        --original_loss \
        --overall_ratio 1.0 \
        --enable_code_switch \
        --code_switch_ratio $CSR \
        --dict_dir $DATA_DIR/dicts \
        --dict_languages ar,bg,de,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh
elif [ $STAGE == 2 ]; then
  FIRST_STAGE_MODEL_PATH="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-CS-csr${CSR}-R1_LAMBDA${R1_LAMBDA}/checkpoint-best"
  OUTPUT_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-CS-csr${CSR}-R1_Lambda${R1_LAMBDA}-Aug1.0-CS-R2_Lambda${R2_LAMBDA}/"
  mkdir -p $OUTPUT_DIR
  python ./src/run_cls.py --model_type xlmr \
        --model_name_or_path $MODEL_PATH \
        --language $LANGS \
        --train_language en \
        --do_train \
        --data_dir $DATA_DIR/$TASK/ \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC \
        --per_gpu_eval_batch_size 64 \
        --learning_rate $LR \
        --num_train_epochs $EPOCH \
        --max_seq_length $MAXL \
        --output_dir $OUTPUT_DIR \
        --task_name $TASK \
        --save_steps -1 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --evaluate_steps $EVALUATE_STEPS \
        --logging_steps 50 \
        --logging_steps_in_sample -1 \
        --logging_each_epoch \
        --gpu_id 0 \
        --seed $SEED \
        --fp16 --fp16_opt_level O2 \
        --warmup_steps -1 \
        --enable_r1_loss \
        --r1_lambda $R1_LAMBDA \
        --original_loss \
        --overall_ratio 1.0 \
        --enable_code_switch \
        --code_switch_ratio $CSR \
        --dict_dir $DATA_DIR/dicts \
        --dict_languages ar,bg,de,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh \
        --first_stage_model_path $FIRST_STAGE_MODEL_PATH \
        --enable_data_augmentation \
        --augment_ratio 1.0 \
        --augment_method cs \
        --r2_lambda $R2_LAMBDA
fi