#!/bin/bash
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1
cwd=${PWD}
src=${PWD}/speechlm/data_process

set -e
mkdir -p dataset/LibriLM/phone_unit/tmp && cd dataset/LibriLM

if [ ! -f librispeech-lm-norm.txt ]; then
    echo "--------------------------------------------------------------------------------------"
    echo "--------Downloading and unpacking librispeech-lm-norm.txt ..."
    echo "--------------------------------------------------------------------------------------"
    wget -c https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
    gzip -d librispeech-lm-norm.txt.gz
fi

# head -1000000 librispeech-lm-norm.txt > phone_unit/tmp/librispeech-lm-norm.txt
cd phone_unit/

echo "--------------------------------------------------------------------------------------"
echo "--------Tokenize the text..."
echo "--------------------------------------------------------------------------------------"
cat ../librispeech-lm-norm.txt | sed '1d' | python $src/wrd2ltr.py > tmp/librilm.ltr

echo "--------------------------------------------------------------------------------------"
echo "--------Tokenize the text to the kaldi-style phonemes ..."
echo "--------------------------------------------------------------------------------------"
python $src/phoneme_tokenizer/ltr2kaldi_phn_sil025.py -i tmp/librilm.ltr -o tmp/librilm
cat tmp/librilm.kaldi_phn_sil025 | sed 's/SIL_S/SIL/g' > tmp/librilm.phn

echo "--------------------------------------------------------------------------------------"
echo "--------Filter too long samples and up-sample phonemes ..."
echo "--------------------------------------------------------------------------------------"
python $src/filter_paireddata_by_len.py -i tmp/librilm -o tmp/librilm_l2k -s phn -t ltr -m 2000
python $src/phoneme_tokenizer/repeat_withou_insert_sil_less_4375.py \
    tmp/librilm_l2k.phn \
    $src/phoneme_tokenizer/mean5_and_std25_sil14_spn32.dict \
    tmp/librilm_l2k_upsample.phn

mv tmp/librilm_l2k.ltr tmp/librilm_l2k_upsample.ltr 
python $src/filter_paireddata_by_len.py -i tmp/librilm_l2k_upsample -o train_text.phn-ltr -s phn -t ltr -m 2800
### the max-length is set to filter the data, considering the batch size (in Large setting, 900,000/320 = 2812 tokens in a batch).


echo "--------------------------------------------------------------------------------------"
echo "--------Create binary files ..."
echo "--------------------------------------------------------------------------------------"
[ ! -f bin-idx/dict.phn.txt ] && echo "dict ${cwd}/dataset/LibriLM/bin-idx/dict.phn.txt not found!" && exit 1
[ ! -f bin-idx/dict.ltr.txt ] && echo "dict ${cwd}/dataset/LibriLM/bin-idx/dict.ltr.txt not found!" && exit 1
bash $src/txt2idx.sh train_text.phn-ltr.phn bin-idx bin-idx/dict.phn.txt
bash $src/txt2idx.sh train_text.phn-ltr.ltr bin-idx bin-idx/dict.ltr.txt

rm -r tmp
cd -
echo "--------------------------------------------------------------------------------------"
echo "--------Done! files are in ${PWD}/dataset/LibriLM"
echo "--------------------------------------------------------------------------------------"
