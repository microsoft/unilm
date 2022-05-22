set -ex
model=$1
batch=$2
beam=$3
inp=$4
bin=$5
out=$6

fairseq-interactive  \
--remove-bpe \
--buffer-size $batch \
--batch-size $batch \
--path $model  \
--max-len-b 200 \
--beam $beam \
--nbest 1 \
--print-alignment \
--source-lang src \
--target-lang trg \
$bin < $inp > $out.tmp

cat $out.tmp | grep "^H"  | cut -f3 >  $out

# --fp16 \