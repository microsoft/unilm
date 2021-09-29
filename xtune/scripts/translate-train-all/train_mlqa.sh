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


cp -r $DATA_DIR/squad/ $DATA_DIR/mlqa/squad1.1/

TASK='mlqa'

TRANSLATION_PATH=$DATA_DIR/xtreme_translations/SQuAD/translate-train/
MODEL_PATH=$DATA_DIR/$MODEL

EPOCH=4
MAXL=384
LANGS="en,es,de,ar,hi,vi,zh"
BSR=0.3
SA=0.3
SNBS=-1
CSR=0.3
R1_LAMBDA=5.0
R2_LAMBDA=0.5
if [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=4
  GRAD_ACC=8
  LR=1.5e-5
else
  BATCH_SIZE=32
  GRAD_ACC=1
  LR=3e-5
fi

if [ $STAGE == 1 ]; then
  OUTPUT_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-CS-csr${CSR}-R1_LAMBDA${R1_LAMBDA}/"
  python ./src/run_qa.py --model_type xlmr \
        --task_name $TASK \
        --model_name_or_path $MODEL_PATH \
        --do_train \
        --do_eval \
        --language $LANGS \
        --train_language en \
        --data_dir $DATA_DIR/$TASK/ \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC \
        --per_gpu_eval_batch_size 128 \
        --learning_rate $LR \
        --num_train_epochs $EPOCH \
        --save_steps 0 \
        --logging_each_epoch \
        --max_seq_length $MAXL \
        --doc_stride 128 \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --evaluate_during_training \
        --logging_steps 50 \
        --evaluate_steps 0 \
        --seed $SEED \
        --fp16 --fp16_opt_level O2 \
        --warmup_steps -1 \
        --enable_r1_loss \
        --r1_lambda $R1_LAMBDA \
        --original_loss \
        --overall_ratio 1.0 \
        --keep_boundary_unchanged \
        --enable_code_switch \
        --code_switch_ratio $CSR \
        --dict_dir $DATA_DIR/dicts \
        --dict_languages es,de,ar,hi,vi,zh \
        --noised_max_seq_length $MAXL
elif [ $STAGE == 2 ]; then
  FIRST_STAGE_MODEL_PATH="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-CS-csr${CSR}-R1_LAMBDA${R1_LAMBDA}/"
  OUTPUT_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-SS-bsr${BSR}-sa${SA}-snbs${SNBS}-R1_Lambda${R1_LAMBDA}-Aug1.0-MT-R2_Lambda${R2_LAMBDA}/"
  python ./src/run_qa.py --model_type xlmr \
        --task_name $TASK \
        --model_name_or_path $MODEL_PATH \
        --do_train \
        --do_eval \
        --language $LANGS \
        --train_language en \
        --data_dir $DATA_DIR/$TASK/ \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC \
        --per_gpu_eval_batch_size 128 \
        --learning_rate $LR \
        --num_train_epochs $EPOCH \
        --save_steps 0 \
        --logging_each_epoch \
        --max_seq_length $MAXL \
        --doc_stride 128 \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --evaluate_during_training \
        --logging_steps 50 \
        --evaluate_steps 0 \
        --seed $SEED \
        --fp16 --fp16_opt_level O2 \
        --warmup_steps -1 \
        --enable_r1_loss \
        --r1_lambda $R1_LAMBDA \
        --original_loss \
        --overall_ratio 1.0 \
        --keep_boundary_unchanged \
        --enable_bpe_sampling \
        --bpe_sampling_ratio $BSR \
        --sampling_alpha $SA \
        --sampling_nbest_size $SNBS \
        --noised_max_seq_length $MAXL \
        --enable_data_augmentation \
        --augment_ratio 1.0 \
        --augment_method mt \
        --translation_path $TRANSLATION_PATH \
        --max_steps 24000 \
        --r2_lambda $R2_LAMBDA \
        --first_stage_model_path $FIRST_STAGE_MODEL_PATH
fi


