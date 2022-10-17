#####################################
# SpeechLM Base model #
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
path_to_lexicon=${DATA_DIR}/librispeech_lexicon.lst
path_to_lm=${DATA_DIR}/4-gram.arpa
[ ! -f $path_to_lexicon ] && echo "Error: $path_to_lexicon not found !" && exit 1
[ ! -f $path_to_lm ] && echo "Error: $path_to_lm not found !" && exit 1

for subset in ${gen_set//,/ }; do
    results_path=$src_dir/decode_${cpt}_ctc/${subset}
    [ ! -d $results_path ] && mkdir -p $results_path

    python $CODE_ROOT/speechlm/infer.py \
    --config-dir $CODE_ROOT/speechlm/config/decode \
    --config-name infer_kenlm \
    common.user_dir=$CODE_ROOT/speechlm \
    \
    dataset.gen_subset=${subset} \
    task.data=$DATA_DIR task.label_dir=$DATA_DIR task.normalize=false \
    common_eval.results_path=${results_path} common_eval.path=${model_path} \
    \
    decoding.lexicon=$path_to_lexicon \
    decoding.lmpath=$path_to_lm \
    decoding.beam=1500 \
    \
    common_eval.quiet=false \
    &
done
wait

### important to know
# When loading the fine-tuned model for decoding, fairseq also loads the pre-trained model to use its states['model'] to build the model instance.
# To prevent the error about the w2v_path (if you don't have the pre-trained model at w2v_path), we set common_eval.model_overrides to override 
# the w2v_path by speechlmp_base_cfg.pt. speechlmp_base_cfg.pt is just a pre-trained model checkpoint without parameters (only contains config).
# So, if you have trained a model with different model config (e.g. different encoder layers), you should modify the common_eval.model_overrides to your own.
    # common_eval.model_overrides=\"{\'w2v_path\':\'$CODE_ROOT/speechlm/config/pretrain/speechlmp_base_cfg.pt\'}\" \
