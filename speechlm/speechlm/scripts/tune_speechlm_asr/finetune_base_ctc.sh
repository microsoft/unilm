# ####################################
# SpeechLM Base model #
# ####################################
[ $# -lt 3 ] && echo "Usage: $0 <model_path> <data_dir> <cpt_tag> [mount=${PWD}] [world_size=8] [update_freq=1]" && exit 1
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1

w2v_path=$1
DATA_DIR=$2
cpt=$3
mount=$4
world_size=$5
update_freq=$6
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=1

CODE_ROOT=${PWD}

exp_name=${w2v_path%/*}
exp_name=${exp_name##*/}
MODEL_DIR="${mount}/exp/finetune_asr/$exp_name/ctc30k_from_${cpt}_bz1.6m_lr1e-5"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/speechlm/config/finetune \
  --config-name speechlm_base_100h \
  common.user_dir=$CODE_ROOT/speechlm \
  \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  model.w2v_path=${w2v_path} \
  \
  optimization.lr=[0.00001] \
  optimization.max_update=30000 \
  dataset.max_tokens=1600000 \
  optimization.update_freq=[${update_freq}] \
  distributed_training.distributed_world_size=${world_size} \
  \
  dataset.train_subset="train_clean_100" \
  dataset.valid_subset="dev_other" \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=${exp_name}

# model_path=/mnt/default/v-ziqzhang/data/speechulm/exp/base/base_speechlmp_32gpu_1accum/checkpoint_298_400000.pt
# data_dir=/home/v-ziqzhang/dataset/LibriSpeech/asr
