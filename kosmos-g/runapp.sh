nohup python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
  app.py None \
  --task kosmosg \
  --criterion kosmosg \
  --arch kosmosg_xl \
  --required-batch-size-multiple 1 \
  --dict-path data/dict.txt \
  --spm-model data/sentencepiece.bpe.model \
  --memory-efficient-fp16 \
  --ddp-backend=no_c10d \
  --distributed-no-spawn \
  --subln \
  --sope-rel-pos \
  --checkpoint-activations \
  --flash-attention \
  --pretrained-ckpt-path /path/to/trained-ckpt > log.log &