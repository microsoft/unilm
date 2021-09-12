set -ex

data_bin=$1
save_dir=$2
PRETRAINED_MODEL=$3

python train.py $data_bin \
    --save-dir $save_dir \
    --arch deltalm_base \
    --pretrained-deltalm-checkpoint $PRETRAINED_MODEL \
    --share-all-embeddings \
    --max-source-positions 512 --max-target-positions 512 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 1e-4 \
    --warmup-init-lr 1e-07 \
    --stop-min-lr 1e-09 \
    --warmup-updates 4000 \
    --max-update 400000 \
    --max-epoch 100 \
    --max-tokens 1024 \
    --update-freq 1 \
    --seed 1 \
    --log-format simple \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe=sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric