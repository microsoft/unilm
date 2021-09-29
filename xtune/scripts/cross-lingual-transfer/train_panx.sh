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

TASK='panx'
MODEL_PATH=$DATA_DIR/$MODEL
EPOCH=10
MAX_LENGTH=128
LANGS="ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu"
EVALUATE_STEPS=1000
BSR=0.3
SA=0.3
SNBS=-1
R1_LAMBDA=5.0
R2_LAMBDA=5.0
if [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=32
  GRAD_ACC=1
  LR=7e-6
else
  BATCH_SIZE=32
  GRAD_ACC=1
  LR=1e-5
fi

TRANSLATION_PATH=$DATA_DIR/xtreme_translations/translate_train.panx.txt

DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/


if [ $STAGE == 1 ]; then
  OUTPUT_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-SS-bsr${BSR}-sa${SA}-snbs${SNBS}-R1_LAMBDA${R1_LAMBDA}/"
  python src/run_tag.py --model_type xlmr \
        --model_name_or_path $MODEL_PATH \
        --do_train \
        --do_eval \
        --do_predict \
        --do_predict_dev \
        --predict_langs $LANGS \
        --train_langs en \
        --data_dir $DATA_DIR \
        --labels $DATA_DIR/labels.txt \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC \
        --per_gpu_eval_batch_size 128 \
        --learning_rate $LR \
        --num_train_epochs $EPOCH \
        --max_seq_length $MAX_LENGTH \
        --noised_max_seq_length $MAX_LENGTH \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --evaluate_during_training \
        --logging_steps 50 \
        --evaluate_steps $EVALUATE_STEPS \
        --seed $SEED \
        --warmup_steps -1 \
        --save_only_best_checkpoint \
        --eval_all_checkpoints \
        --eval_patience -1 \
        --fp16 --fp16_opt_level O2 \
        --hidden_dropout_prob 0.1 \
        --original_loss \
        --enable_r1_loss \
        --r1_lambda $R1_LAMBDA \
        --use_token_label_probs \
        --enable_bpe_sampling \
        --bpe_sampling_ratio $BSR \
        --sampling_alpha $SA \
        --sampling_nbest_size $SNBS
elif [ $STAGE == 2 ]; then
  FIRST_STAGE_MODEL_PATH="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-SS-bsr${BSR}-sa${SA}-snbs${SNBS}-R1_LAMBDA${R1_LAMBDA}/checkpoint-best"
  OUTPUT_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-SS-bsr${BSR}-sa${SA}-snbs${SNBS}-R1_Lambda${R1_LAMBDA}-Aug1.0-SS-R2_Lambda${R2_LAMBDA}/"
  python src/run_tag.py --model_type xlmr \
        --model_name_or_path $MODEL_PATH \
        --do_train \
        --do_eval \
        --do_predict \
        --do_predict_dev \
        --predict_langs $LANGS \
        --train_langs en \
        --data_dir $DATA_DIR \
        --labels $DATA_DIR/labels.txt \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC \
        --per_gpu_eval_batch_size 128 \
        --learning_rate $LR \
        --num_train_epochs $EPOCH \
        --max_seq_length $MAX_LENGTH \
        --noised_max_seq_length $MAX_LENGTH \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --evaluate_during_training \
        --logging_steps 50 \
        --evaluate_steps $EVALUATE_STEPS \
        --seed $SEED \
        --warmup_steps -1 \
        --save_only_best_checkpoint \
        --eval_all_checkpoints \
        --eval_patience -1 \
        --fp16 --fp16_opt_level O2 \
        --hidden_dropout_prob 0.1 \
        --original_loss \
        --enable_r1_loss \
        --r1_lambda $R1_LAMBDA \
        --use_token_label_probs \
        --enable_bpe_sampling \
        --bpe_sampling_ratio $BSR \
        --sampling_alpha $SA \
        --sampling_nbest_size $SNBS \
        --enable_data_augmentation \
        --augment_ratio 1.0 \
        --augment_method ss \
        --r2_lambda $R2_LAMBDA \
        --first_stage_model_path $FIRST_STAGE_MODEL_PATH \
        --use_hard_labels
fi