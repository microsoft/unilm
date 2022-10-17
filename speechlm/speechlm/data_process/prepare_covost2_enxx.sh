
#!/bin/bash
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1
[ $# -lt 1 ] && echo "Usage: $0 <lang> [root=${PWD}/dataset/CommonVoice/v4]" && exit 0
cwd=${PWD}
src=${PWD}/speechlm/data_process
lang=$1
root=$2
[ -z $root ] && root="${PWD}/dataset/CommonVoice/v4"
set -e -o pipefail -u


### step1, convert mp3 to wav
cd $root/en && mkdir -p wav
cut -f2 validated.tsv | sed '1d' | sed "s|^|${root}/en/clips/|" > validated.id
for i in $(seq 0 39); do 
        echo extracting $i; 
        python $src/covost2/mp3_to_wav.py -i validated.id -n 40 -r $i &
done
wait
cd $cwd


### step2, manifest
datadir="$root/en/en-$lang" && mkdir -p $datadir && cd $datadir
python /mnt/default/v-ziqzhang/code/stpretrain_scripts/data_process/covost2/prepare_covost_data.py --data-root $root --src-lang en --tgt-lang $lang --vocab-type char
mv ../*en_${lang}.* ./

# adjust config_base_en${lang}.yaml
echo "bpe_tokenizer:" > config_base_en${lang}.yaml
echo "  bpe: sentencepiece" >> config_base_en${lang}.yaml
echo "  sentencepiece_model: spm_char_st_en_de.model" >> config_base_en${lang}.yaml
echo "" >> config_base_en${lang}.yaml
echo "shuffle: false" >> config_base_en${lang}.yaml
echo "use_audio_input: true" >> config_base_en${lang}.yaml
echo "use_sample_rate: 16000" >> config_base_en${lang}.yaml
echo "standardize_audio: false" >> config_base_en${lang}.yaml
echo "vocab_filename: spm_char_st_en_de.txt" >> config_base_en${lang}.yaml
echo "" >> config_base_en${lang}.yaml
echo "# required by speech_to_text task but never used" >> config_base_en${lang}.yaml
echo "input_channels: 1" >> config_base_en${lang}.yaml
echo "input_feat_per_channel: 1" >> config_base_en${lang}.yaml
echo "" >> config_base_en${lang}.yaml
# adjust config_large_en${lang}.yaml
cat config_base_en${lang}.yaml | sed "s|standardize_audio: false|standardize_audio: true|" > config_large_en${lang}.yaml
