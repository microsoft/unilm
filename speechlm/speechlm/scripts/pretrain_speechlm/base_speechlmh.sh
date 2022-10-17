# ####################################
# SpeechLM-H Base model #
# ####################################
[ $# -lt 2 ] && echo "Usage: $0 <data_dir> <text_data_dir> [mount=${PWD}] [world_size=32] [update_freq=1]" && exit 1
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1
DATA_DIR=$1
TEXT_DATA_DIR=$2
mount=$3
world_size=$4
update_freq=$5
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=32
[ -z $update_freq ] && update_freq=1

CODE_ROOT=${PWD}
MODEL_DIR="${mount}/exp/pretrain/base_speechlmh_${world_size}gpu_${update_freq}accum"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/speechlm/config/pretrain \
  --config-name speechlm_base_librispeech \
  common.user_dir=$CODE_ROOT/speechlm \
  \
  task.labels='["km"]' \
  model.label_rate=50 \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  task.text_cfg.text_data=$TEXT_DATA_DIR \
  \
  dataset.train_subset=\"train_960+train_text.km-ltr\" \
  dataset.valid_subset=\"dev_clean+dev_clean.km-ltr\" \
  dataset.num_workers=0 \
  dataset.max_tokens=1400000 \
  distributed_training.distributed_world_size=${world_size} \
  optimization.update_freq=[${update_freq}] \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=pretrain

# data_dir="/mnt/default/v-ziqzhang/data/stbert/data/librispeech/hubert_release_iter2_layer9_kmeans/local"
# text_data_dir="/mnt/default/v-ziqzhang/dataset/LibriLM/from_fastT2U/bin-idx"
