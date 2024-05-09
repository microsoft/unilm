TASK='harness_boolq'
# TASK='hendrycksTest-abstract_algebra'

cd yoco/
torchrun --master-port=29505 --nproc_per_node=1 validate.py \
    --data-dir ../harness_data/ \
    --criterion harness_eval \
    --task harness_eval \
    --batch-size 4 \
    --eval-data ${TASK}  \
    --log-format simple  --log-interval 10 \
    --bf16 \
    --tokenizer-pad-to-multiple 8 \
    --arch yoco_3b_new --tiktoken-model cl100k_base --load-ckpt /data/yutao/ckpt_opensource/YOCO-3B-1M/checkpoint.pth --yoco-model /data/yutao/ckpt_opensource/YOCO-3B-1M  --tokens-per-sample 4096
    # --arch llama_from_ckpt --llama-model /data/yutao/llama/llama-2-7b --load-ckpt /data/yutao/llama/llama-2-7b/consolidated.00.pth --tokens-per-sample 4096


