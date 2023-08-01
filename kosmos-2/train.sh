#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=50000 train.py /path/to/text config/    \
        --task image_gpt_interleaved_laion_obj \
        --tokens-per-sample 2048 \
        --criterion unigpt \
        --arch unigptmodel_xl      \
        --required-batch-size-multiple 1      \
        --optimizer adam       \
        --adam-betas '(0.9,0.98)'       \
        --adam-eps 1e-6       \
        --clip-norm 2.0      \
        --lr-scheduler polynomial_decay       \
        --weight-decay 0.01       \
        --lr 0.0002       \
        --warmup-updates 375       \
        --total-num-update 60000      \
        --max-update 60000      \
        --max-sentences 1      \
        --update-freq 8      \
        --log-format simple      --log-interval 50     --disable-validation      \
        --save-interval-updates 5000     --no-epoch-checkpoints      \
        --memory-efficient-fp16     --fp16-init-scale 4     --fp16-scale-window 256      \
        --min-loss-scale 0.0001      \
        --seed 2      \
        --dict-path data/dict.txt       \
        --spm-model data/sentencepiece.bpe.model      \
        --save-dir ./output/debug \
        --tensorboard-logdir ./output/debug/tb-logs      \
        --init-from-file /path/to/init file \
        --ddp-backend=no_c10d      \
        --distributed-no-spawn      \
        --batch-read-ahead 100       \
        --reset-dataloader  \
        --train-json-split-name train-nogithub-noarvix-nopubmed-mtnlg \
        --image-encoder clip   --visual-model-name ViT-L-14 --visual-output-dim  1024 \
        --visual-pretrained /path/to/openai_clip/ViT-L-14-sd.pt   \
        --interleaved-data-dir /path/to/interleaved data \
        --interleaved-batch-size 1 \
        --laion-data-dir /path/to/laion_config \
        --laion-batch-size 6 \
        --phrase-mode 'expression' \
        --quantized-size 32 \
        --locate-special-token 1 \
        --box-score-threshold 0.65 \
        --mix-no-object-prob 0. \
        --use-object-bbox-prob 0.5 \
        --latent-query-num 64 --connector xconnector \
        --no-freeze-all \
        --subln  --flash-attention  --sope-rel-pos \
        --data-weights 0,8,0 \
        --checkpoint-activations 

### Instruction of some args

# /path/to/text config/ Path to Text data config; if no, 'None' is OK.
# task: image_gpt_interleaved_laion_obj task is defined in [here](unilm/tasks/gpt_interleaved_laion_obj.py).
# arch: unigptmodel_xl model is defined in [here](unilm/models/unigpt.py).
# total-num-update & max-update: Training steps.
# update-freq: Gradient update frequency.
# init-from-file: Init from the given file. if no, just remove this arg.
# visual-pretrained: Model ckpt of clip_l/14, you can download from openclip.
# interleaved-data-dir: Path to interleaved data. Due to the policy, we can't release it. But the open-source mmc4 is also a good alternative.
# laion-data-dir: Path to laion or coyo dataset.
# phrase-mode: use the referring expressions or phrases to train the model. Refer to [here](unilm/data/vl/obj_utils.py) for details.
# quantized-size: size of each quantized bucket, used for discretizing continuous coordinates.
# locate-special-token: use special token (<grounding>) to prompt model to generate bounding boxes.
# mix-no-object-prob: probability of using the image-text pairs without boxes.
# use-object-bbox-prob: probability of using the image-text pairs with boxes.
# latent-query-num: Number of image tokens for language model.
# connector: resampler type, see [here](unilm/models/connector.py).
# no-freeze-all: learn all parameters.
# data-weights: sample probability of interleaved data, laion, text
