cd yoco/
torchrun --master-port=29504 --nproc_per_node=1 validate.py \
    --task pseudo \
    --criterion multi_needle --needle-num 4 \
    --batch-size 1 \
    --max-epoch 1 \
    --no-save \
    --tiktoken-model cl100k_base \
    --bf16 \
    --arch yoco_3b_new --tiktoken-model cl100k_base --load-ckpt /data/yutao/ckpt_opensource/YOCO-3B-1M/checkpoint.pth --yoco-model /data/yutao/ckpt_opensource/YOCO-3B-1M --tokens-per-sample 1048576 --interval 1048576

