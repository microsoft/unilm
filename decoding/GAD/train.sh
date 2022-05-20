#!/bin/bash

bin_path=./wmt14.en-de-binary-file
lr=0.0005
dropout=0.1
warmup=10000
size=25
seed=1
max_tokens=4096
update_freq=4
update=300000

python train.py ${bin_path} --arch block \
    --noise block_mask --share-all-embeddings --criterion glat_loss --label-smoothing 0.1 \
    --lr ${lr} --warmup-init-lr 1e-7 --stop-min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates ${warmup} \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_lev_modified --max-tokens ${max_tokens} \
    --weight-decay 0.01 --dropout ${dropout} --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 \
    --decoder-embed-dim 512 --fp16 --max-source-positions 1000 --max-target-positions 1000 --max-update ${update}\
    --seed ${seed} --clip-norm 5 --save-dir ./checkpoints \
    --src-embedding-copy --log-interval 1000 --user-dir block_plugins --block-size ${size} --total-up ${update} \
    --update-freq ${update_freq} --decoder-learned-pos --encoder-learned-pos --apply-bert-init --activation-fn gelu