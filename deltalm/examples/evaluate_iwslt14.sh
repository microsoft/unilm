set -ex

data_bin=$1
save_dir=$2

python generate.py $data_bin \
    --path $save_dir/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe=sentencepiece