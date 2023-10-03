# Example: Integration with FairSeq

## Setup

```bash
# Install the repo as a package:
git clone https://github.com/msranlp/torchscale.git
cd torchscale
pip install -e .
pip install git+https://github.com/shumingma/fairseq.git@moe
pip install git+https://github.com/shumingma/infinibatch.git
pip install iopath
pip install --upgrade numpy
```

## Example: BERT Pretraining

### Data Format

We use a [streaming dataloader](https://github.com/microsoft/infinibatch) to read the data on-the-fly from the disk. It requires the data sharded into multiple small files (e.g. 10K lines per file), as well as a JSON file to contain some meta data and the paths to these files.

The overall data directory should be organized as follows:
```
Data/
├── json/
│   ├── train.json
│   └── valid.json
├── shard/
│   ├── train/
│   │   ├── 00000.txt
│   │   ├── 00001.txt
│   │   └── ...
│   └── valid/
│       ├── 00000.txt
│       ├── 00001.txt
│       └── ...
├── dict.txt
└── sentencepiece.bpe.model
```

We recommend that each sharded data files contains no more than 10K lines with one sentence per line, and two documents should be separated with an empty line.
```
Document 1 Line 1
Document 1 Line 2
Document 1 Line 3

Document 2 Line 1
Document 2 Line 2

...
```

Also, the JSON file should be in the format like this:
```
[
    {
        "source": [
            "shard/train/00000.txt",
            "shard/train/00001.txt",
            ...
        ],
        "source_lang": "en",
        "weight": 1.0
    }
]
```

### Training Command
```bash
cd examples/fairseq/
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 train.py ${PATH_TO_DATA} \
        --task pretraining  \
        --tokens-per-sample 512  \
        --mask-prob 0.15  \
        --span-length 3.0  \
        --leave-unmasked-prob 0.0  \
        --random-token-prob 0.0 \
        --criterion masked_lm  \
        --arch mlm_base  \
        --share-encoder-input-output-embed \
        --required-batch-size-multiple 8 \
        --spm-model ${PATH_TO_DATA}/sentencepiece.bpe.model \
        --dict-file ${PATH_TO_DATA}/dict.txt \
        --optimizer adam  \
        --adam-betas '(0.9,0.98)'  \
        --adam-eps 1e-6  \
        --clip-norm 2.0 \
        --lr-scheduler polynomial_decay  \
        --lr 0.0005  \
        --warmup-updates 10000  \
        --total-num-update 125000 \
        --max-update 125000 \
        --max-sentences 32  \
        --update-freq 1 \
        --log-format simple  \
        --log-interval 100 \
        --disable-validation \
        --save-interval-updates 5000 \
        --no-epoch-checkpoints \
        --fp16 \
        --fp16-init-scale 4 \
        --fp16-scale-window 256 \
        --min-loss-scale 0.0001 \
        --seed 1 \
        --save-dir ${PATH_TO_CKPT} \
        --ddp-backend=no_c10d \
        --distributed-no-spawn \
        --reset-dataloader \
        --batch-read-ahead 10000 \
        --rel-pos-buckets 32 \
        --max-rel-pos 128 \
        --deepnorm
```

## Example: GPT Pretraining

### Data Format

We use the format as in the FairSeq's [language modeling example](https://github.com/facebookresearch/fairseq/tree/main/examples/language_model#1-preprocess-the-data).

### Dense Model

```bash
cd examples/fairseq/
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py \
    ${PATH_TO_DATA} \
    --num-workers 2 \
    --activation-fn gelu \
    --share-decoder-input-output-embed \
    --validate-interval-updates 1000 \
    --save-interval-updates 1000 \
    --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --fp16-init-scale 4 \
    --arch lm_base \
    --task language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 128 \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 750 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size 4 \
    --update-freq 1 \
    --required-batch-size-multiple 1 \
    --total-num-update 50000 \
    --max-update 50000 \
    --seed 1 \
    --ddp-backend=c10d
```

### Sparse (MoE) Model

```bash
cd examples/fairseq/
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py \
    ${PATH_TO_DATA} \
    --num-workers 2 \
    --activation-fn gelu \
    --share-decoder-input-output-embed \
    --validate-interval-updates 1000 \
    --save-interval-updates 1000 \
    --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --fp16-init-scale 4 \
    --arch lm_base \
    --task language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 128 \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 750 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size 4 \
    --update-freq 1 \
    --required-batch-size-multiple 1 \
    --total-num-update 50000 \
    --max-update 50000 \
    --seed 1 \
    --ddp-backend=no_c10d \
    --moe-expert-count 2 --moe-freq 2 \
    --moe-gating-use-fp32 --moe-second-expert-policy random --moe-normalize-gate-prob-before-dropping \
    --moe-eval-capacity-token-fraction -1.0 \
    --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
    --use-xmoe
```

## Example: Machine Translation

### Data Format

We follow the FairSeq's [neural machine translation example](https://github.com/facebookresearch/fairseq/tree/main/examples/translation#training-a-new-model) to preprocess the data.

### Dense Model

```bash
cd examples/fairseq/
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py \
    ${PATH_TO_DATA} \
    --arch mt_base --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens 4096 --fp16
```

### Sparse (MoE) Model

```bash
cd examples/fairseq/
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py \
    ${PATH_TO_DATA} \
    --arch mt_base --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --moe-expert-count 2 --moe-freq 2 \
    --moe-gating-use-fp32 --moe-second-expert-policy random --moe-normalize-gate-prob-before-dropping \
    --moe-eval-capacity-token-fraction -1.0 \
    --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
    --use-xmoe \
    --max-tokens 4096 --fp16
```
