# ####################################
# SpeechLM-P Large model #
# ####################################
[ $# -lt 2 ] && echo "Usage: $0 <data_dir> <text_data_dir> [mount=${PWD}] [world_size=32] [update_freq=4]" && exit 1
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1
DATA_DIR=$1
TEXT_DATA_DIR=$2
mount=$3
world_size=$4
update_freq=$5
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=32
[ -z $update_freq ] && update_freq=4

CODE_ROOT=${PWD}
MODEL_DIR="${mount}/exp/pretrain/large_speechlmp_${world_size}gpu_${update_freq}accum"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/speechlm/config/pretrain \
  --config-name speechlm_large_librilight \
  common.user_dir=$CODE_ROOT/speechlm \
  \
  task.labels='["phn"]' \
  model.label_rate=50 \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  task.text_cfg.text_data=$TEXT_DATA_DIR \
  \
  dataset.train_subset=\"train_60k+train_text.phn-ltr\" \
  dataset.valid_subset=\"dev_clean+dev_clean.phn-ltr\" \
  dataset.num_workers=1 \
  dataset.max_tokens=900000 \
  distributed_training.distributed_world_size=${world_size} \
  optimization.update_freq=[${update_freq}] \
  \
  common.fp16_scale_tolerance=0.1 \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=pretrain

# data_dir="/stdblob/users/v-ziqzhang/dataset/librilight/chunkdata"
# text_data_dir="/stdblob/users/v-ziqzhang/dataset/LibriLM/phn2char_sanych/filt2k_sil025_m5std25_sil14_spn32/bin-idx"
