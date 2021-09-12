set -ex

input_dir=$1
output_dir=$2

SPM_MODEL=$3

mkdir -p $output_dir

for split in train valid test ;
do
    for lang in en de ;
    do
        cat $input_dir/$split.$lang | spm_encode --model=$SPM_MODEL --output_format=piece > $output_dir/$split.$lang;
    done;
done;