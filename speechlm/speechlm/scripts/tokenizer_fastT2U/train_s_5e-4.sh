#####################################
# Fast Text2Unit Model #
#####################################
[ $# -lt 1 ] && echo "Usage: $0 <data_dir> [mount] [world_size=4] [update_freq=1]" && exit 0
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1

DATA_DIR=$1
mount=$2
world_size=$3
update_freq=$4
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=4
[ -z $update_freq ] && update_freq=1

CODE_ROOT=${PWD}
MODEL_DIR="$mount/exp/fast_text2unit/small_lr5e-4_tristage_ls0.1_${world_size}gpu_${update_freq}accum"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

fairseq-train ${DATA_DIR} --save-dir ${MODEL_DIR} \
  --config-yaml config.yaml \
  --user-dir $CODE_ROOT/speechlm \
  --train-subset train_100 --valid-subset dev_clean \
  --num-workers 4 --max-tokens 20000 \
  --distributed-world-size ${world_size} --update-freq ${update_freq} \
  \
  --task fast_text_to_unit --criterion fasttext2unit_criterion --arch fasttext2unit_s \
  --label-smoothing 0.1 \
  \
  --clip-norm 5.0 --n-frames-per-step 1 \
  --dropout 0.1 --attention-dropout 0.1 \
  --optimizer adam --lr 5e-4 --lr-scheduler tri_stage --phase-ratio [0.3,0.0,0.7] --max-update 10000 \
  --seed 1 --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
  \
  --save-interval 2 \
  --tensorboard-logdir ${MODEL_DIR} \
  --fp16 --find-unused-parameters \
  | tee ${MODEL_DIR}/train.log

# DATA_DIR=/mnt/default/v-ziqzhang/dataset/librispeech_phone2unit/phone2unit
