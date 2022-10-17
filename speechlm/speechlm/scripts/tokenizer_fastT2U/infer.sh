#####################################
# Fast Text2Unit Model #
#####################################
[ $# -lt 2 ] && echo "Usage: $0 <model_path> <gen_set> " && exit 0
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1

model_path=$1
src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

gen_set=$2

DATA_DIR=${gen_set%/*}
gen_set=${gen_set##*/}
outdir=$src_dir/decode_${cpt}

CODE_ROOT=${PWD}

for subset in ${gen_set//,/ }; do
    results_path=$outdir/phone2unit_${subset}
    [ ! -d $results_path ] && mkdir -p $results_path

    python $CODE_ROOT/speechlm/generate_unit.py $DATA_DIR \
    --user-dir $CODE_ROOT/speechlm \
    --config-yaml config.yaml \
    --path ${model_path} \
    --task fast_text_to_unit \
    --gen-subset $subset \
    \
    --beam 1 \
    --max-tokens 10000 \
    --results-path $results_path \
    --scoring sacrebleu

    echo $results_path
    tail -n 1 $results_path/generate-*.txt
    sleep 1s
done

# --distributed-world-size 1000 --distributed-rank 0 \
