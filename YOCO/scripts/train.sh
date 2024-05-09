cd yoco/
torchrun --master-port=29501 --nproc-per-node=1 train.py /mnt/nlcredstone/shaohanh/data/redstone_v4_21_config \
    --save-interval-updates 5000 \
    --no-epoch-checkpoints \
    --arch yoco_base \
    --criterion cross_entropy \
    --task gpt \
    --tokens-per-sample 2048 \
    --tokenizer-pad-to-multiple 8 \
    --pad-to-max-len \
    --optimizer adam --adam-betas "(0.9, 0.95)" \
    --adam-eps 1e-06 \
    --clip-norm 2.0 \
    --lr 0.00015 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 50 \
    --weight-decay 0.05 \
    --batch-size 1  \
    --model-parallel-size 1 \
    --update-freq 1 \
    --batch-read-ahead 1000 \
    --total-num-update 300000 \
    --log-format simple      --log-interval 10    --disable-validation \
    --tiktoken-model cl100k_base \
    --no-save \
    --bf16 \

