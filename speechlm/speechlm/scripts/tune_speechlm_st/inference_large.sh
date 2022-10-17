# ####################################
# SpeechLM Base model #
# ####################################
[ $# -lt 3 ] && echo "Usage: $0 <model_path> <data_dir> <lang> [gen-set=dev] [beam_size=5] [lenpen=1.0]" && exit 0
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1

model_path=$1
DATA_DIR=$2
lang=$3
gen_set=$4
beam_size=$5
lenpen=$6
[ -z $gen_set ] && gen_set="dev"
[ -z $beam_size ] && beam_size=5
[ -z $lenpen ] && lenpen=1
src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

CODE_ROOT=${PWD}
results_path=$src_dir/decode_${cpt}_beam${beam_size}/${gen_set}
[ ! -d $results_path ] && mkdir -p $results_path

python $CODE_ROOT/fairseq/fairseq_cli/generate.py $DATA_DIR \
    --gen-subset ${gen_set}_st_en_${lang}_local \
    --max-tokens 2300000 \
    --max-source-positions 2300000 \
    --num-workers 0 \
    \
    --user-dir $CODE_ROOT/speechlm \
    --task speech_to_text \
    --config-yaml config_large_en${lang}.yaml \
    \
    --path ${model_path} \
    --results-path $results_path \
    \
    --scoring sacrebleu --max-len-a 0 --max-len-b 512 \
    --beam ${beam_size} \
    --lenpen $lenpen \

    echo $results_path
    tail -n 1 $results_path/generate-*.txt
    sleep 1s

# model_path=/mnt/default/v-ziqzhang/data/speechulm/finetune_covost/large_speechlmp_32gpu_4accum/legacy_ende_from_400k_bz3.6m_lr1e-4/checkpoint.avgnbest_convert.pt
# data_dir=dataset/CommonVoice/v4/en/en-de
