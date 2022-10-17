#!/bin/bash
[ $# -lt 3 ] && echo "Usage: $0 <input-text> <outdir> <DICT> <suffix>" && exit 0

input=$1
outdir=$2
DICT=$3
suffix=$4
outname=${input##*/}
outname=${outname%.txt*}
[ -z $input ] && echo "You must specify a source file" && exit 1

[ -z $DICT ] && echo "No dict was specified!" && exit 1
[ -z $outdir ] && outdir=${input%/*}
[ -z $outdir ] && outdir="."
[ ! -d $outdir ] && mkdir -p $outdir

echo "------------------------------- creating idx/bin--------------------------------------------"
echo "$input --> $outdir/${outname}${suffix}.idx"
fairseq-preprocess \
  --only-source \
  --trainpref $input \
  --destdir $outdir \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --workers 40

mv $outdir/train.idx $outdir/${outname}${suffix}.idx
mv $outdir/train.bin $outdir/${outname}${suffix}.bin
echo "-----------------------------------   done      --------------------------------------------"

