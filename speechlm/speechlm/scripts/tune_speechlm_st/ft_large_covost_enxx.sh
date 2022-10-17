# ####################################
# SpeechLM Large model #
# ####################################
[ $# -lt 4 ] && echo "Usage: $0 <model_path> <data_dir> <lang> <cpt-tag> [mount=${PWD}] [world_size=8] [update_freq=4]" && exit 0
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1

w2v_path=$1
DATA_DIR=$2
lang=$3
cpt=$4
mount=$5
world_size=$6
update_freq=$7
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=4

CODE_ROOT=${PWD}

exp_name=${w2v_path%/*}
exp_name=${exp_name##*/}
MODEL_DIR="$mount/exp/finetune_covost/$exp_name/legacy_en${lang}_from_${cpt}_bz3.6m_lr1e-4"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

max_tokens=900000
python $CODE_ROOT/fairseq/fairseq_cli/train.py ${DATA_DIR} \
    --save-dir ${MODEL_DIR} \
    --user-dir $CODE_ROOT/speechlm \
    --task speech_to_text \
    --config-yaml config_large_en${lang}.yaml \
    --train-subset "train_st_en_${lang}_local" \
    --valid-subset "dev_st_en_${lang}_local" \
    --fp16 \
    --seed 1 \
    \
    --ddp-backend no_c10d \
    --distributed-world-size ${world_size} \
    --tensorboard-logdir ${MODEL_DIR} \
    \
    --criterion label_smoothed_cross_entropy --report-accuracy \
    --label-smoothing 0.1 \
    \
    --optimizer adam \
    --clip-norm 1.0 \
    --lr 1e-04 \
    --lr-scheduler polynomial_decay --warmup-updates 5000 \
    --max-update 50000 \
    --total-num-update 50000 \
    --update-freq ${update_freq} \
    \
    --max-tokens ${max_tokens} \
    --max-sentences 16 \
    --max-tokens-valid ${max_tokens} \
    --grouped-shuffling \
    --max-source-positions ${max_tokens} \
    --skip-invalid-size-inputs-valid-test \
    --num-workers 0 \
    --best-checkpoint-metric "accuracy" \
    --maximize-best-checkpoint-metric \
    \
    --arch "speechlm_st_legacy" \
    --w2v-path ${w2v_path} --encoder-embed-dim 1024 \
    --layerdrop 0.1 \
    --decoder-layerdrop 0.1 \
    --activation-dropout 0.0 \
    --attention-dropout 0.1 \
    --feature-grad-mult 1.0 \
    \
    --apply-mask --mask-prob 0.5 \
    \
    --log-format json \
    --log-interval 100 \
    --save-interval 1 \
    --keep-last-epochs 5 \
    --keep-best-checkpoints 5 \
    \
    2>&1 | tee ${MODEL_DIR}/train.log

# model_path=/mnt/default/v-ziqzhang/data/speechulm/exp/large/large_speechlmp_32gpu_4accum/checkpoint_31_400000.pt
# data_dir=${HOME}/dataset/CommonVoice/v4/en/en-de
