#!/bin/bash

GEN=./output/beam5_result_en_de.out # a fairseq standard output file
OUT=./output/block.out # GAD output file

SYS=./output/block.out.compound.sys
REF=./data/test.de.compound.ref

grep ^T $GEN \
| sed 's/^T\-//' \
| sort -n -k 1  \
| cut -f 2 \
| perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' \
> $REF

cat $OUT \
| perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' \
> $SYS

fairseq-score --sys $SYS --ref $REF