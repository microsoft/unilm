set -ex

input_dir=$1
output_dir=$2
dict_file=$3

python preprocess.py  \
    --trainpref $input_dir/train \
    --validpref $input_dir/valid \
    --testpref $input_dir/test \
    --source-lang de --target-lang en \
    --destdir $output_dir \
    --srcdict $dict_file \
    --tgtdict $dict_file \
    --workers 40