#####################################
# SpeechLM Large model #
#####################################
[ $# -lt 2 ] && echo "Usage: $0 <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]" && exit 1
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1

model_path=$1
DATA_DIR=$2
gen_set=$3
[ -z $gen_set ] && gen_set="dev_clean,dev_other,test_clean,test_other"
src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

CODE_ROOT=${PWD}

for subset in ${gen_set//,/ }; do
    results_path=$src_dir/decode_${cpt}_ctc/${subset}
    [ ! -d $results_path ] && mkdir -p $results_path

    python $CODE_ROOT/speechlm/infer.py \
    --config-dir $CODE_ROOT/speechlm/config/decode \
    --config-name infer_viterbi \
    common.user_dir=$CODE_ROOT/speechlm \
    \
    dataset.gen_subset=${subset} \
    task.data=$DATA_DIR task.label_dir=$DATA_DIR task.normalize=true \
    common_eval.results_path=${results_path} common_eval.path=${model_path} \
    \
    common_eval.quiet=true \
    &
done
wait

# model_path=/mnt/default/v-ziqzhang/data/speechulm/finetune_asr/large_speechlmp_32gpu_4accum/ctc200k_from_400k_bz3.6m_lr1e-5/checkpoint_convert.pt
# data_dir=/home/v-ziqzhang/dataset/LibriSpeech/asr
