# Fine-tuning BEiT on ImageNet-1k (image classification)

## Setup

1. [Setup environment](README.md#setup).
2. Download and extract ImageNet-1k from http://image-net.org/.

The directory structure is the standard layout of torchvision's [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder). The training and validation data are expected to be in the `train/` folder and `val` folder, respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


## Fine-tuning

We recommend you to use the checkpoints that are **self-supervised pretrained and then intermediate fine-tuned** on ImageNet-22k for better performance. We use following commands to fine-tune BEiT-large with 8 V100-16GB cards:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model beit_large_patch16_224 --data_path /path/to/imagenet \
    --finetune https://github.com/addf400/files/releases/download/v1.0/beit_large_patch16_224_pt22k_ft22k.pth \
    --output_dir /path/to/save_result --batch_size 32 --lr 2e-5 --update_freq 2 \
    --warmup_epochs 5 --epochs 30 --layer_decay 0.9 --drop_path 0.4 \
    --weight_decay 1e-8 --enable_deepspeed
```
- `--batch_size`: batch size per GPU.
- `--update_freq`: gradient accumulation steps.
- Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*32*2 = 512`. The three arguments need to be adjusted together in order to keep the total batch size unchanged.
- Gradient accumulation: if your GPU memory is limited (i.e., OOM issues), you can reduce `--batch size` and increase `--update_freq` to use the same effective batch size.
- `--enable_deepspeed`: enable [deepspeed](https://github.com/microsoft/DeepSpeed) during fine-tuning, which uses Apex O2 mixed-precision training. If without this argument, the code will use torch.amp for fine-tuning.
- In order to fine-tune BEiT in higher resolution (such as 384), we can set `--input_size 384` and reset `--model beit_large_patch16_384`. Gradient accumulation can be used for OOM issues.

For BEiT-base, we set `--layer_decay 0.85 --drop_path 0.1` and keep other arguments unchanged as follows:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model beit_base_patch16_224 --data_path /path/to/imagenet \
    --finetune https://github.com/addf400/files/releases/download/v1.0/beit_base_patch16_224_pt22k_ft22k.pth \
    --output_dir /path/to/save_result --batch_size 64 --lr 2e-5 --update_freq 1 \
    --warmup_epochs 5 --epochs 30 --layer_decay 0.85 --drop_path 0.1 \
    --weight_decay 1e-8 --enable_deepspeed
```

For the BEiT models that are fully self-supervised pretrained on ImageNet-22k (without intermediate fine-tuning on ImageNet-22k), we recommend you to enable mixup and cutmix during fine-tuning. We use the following commands for BEiT-large and BEiT-base:

```bash
# BEiT-large
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model beit_large_patch16_224 --data_path /path/to/imagenet \
    --finetune https://github.com/addf400/files/releases/download/v1.0/beit_large_patch16_224_pt22k.pth \
    --output_dir /path/to/save_result --batch_size 64 --lr 1e-3 --update_freq 2 \
    --warmup_epochs 5 --epochs 50 --layer_decay 0.75 --drop_path 0.2 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 --enable_deepspeed
# BEiT-base
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model beit_base_patch16_224 --data_path /path/to/imagenet \
    --finetune https://github.com/addf400/files/releases/download/v1.0/beit_base_patch16_224_pt22k.pth \
    --output_dir /path/to/save_result --batch_size 128 --lr 4e-3 --update_freq 1 \
    --warmup_epochs 20 --epochs 100 --layer_decay 0.65 --drop_path 0.1 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 --enable_deepspeed
```


## Evaluate our fine-tuned checkpoints

- Evaluate our fine-tuned BEiT-large model in 224 resolution on ImageNet val with a single GPU:
```bash
python run_class_finetuning.py \
    --eval --model beit_large_patch16_224 --data_path /path/to/imagenet \
    --resume https://github.com/addf400/files/releases/download/v1.0/beit_large_patch16_224_pt22k_ft22kto1k.pth 
```
Expected results:
```
* Acc@1 87.396 Acc@5 98.282 loss 0.515
```

- Evaluate our fine-tuned BEiT-base model in 384 resolution on ImageNet val with a single GPU:
```bash
python run_class_finetuning.py \
    --eval --model beit_base_patch16_384 --input_size 384 --data_path /path/to/imagenet \
    --resume https://github.com/addf400/files/releases/download/v1.0/beit_base_patch16_384_pt22k_ft22kto1k.pth 
```
Expected results:
```
* Acc@1 86.820 Acc@5 98.124 loss 0.565
```

- Evaluate our fine-tuned BEiT-large model in 384 resolution on ImageNet val with a single GPU:
```bash
python run_class_finetuning.py \
    --eval --model beit_large_patch16_384 --input_size 384 --data_path /path/to/imagenet \
    --resume https://github.com/addf400/files/releases/download/v1.0/beit_large_patch16_384_pt22k_ft22kto1k.pth 
```
Expected results:
```
* Acc@1 88.408 Acc@5 98.602 loss 0.479
```

- Evaluate our fine-tuned BEiT-large model in 512 resolution on ImageNet val with a single GPU:
```bash
python run_class_finetuning.py \
    --eval --model beit_large_patch16_512 --input_size 512 --data_path /path/to/imagenet \
    --resume https://github.com/addf400/files/releases/download/v1.0/beit_large_patch16_512_pt22k_ft22kto1k.pth 
```
Expected results:
```
* Acc@1 88.600 Acc@5 98.658 loss 0.474
```

## Example: Linear probe on ImageNet

To train supervised linear classifier on frozen weights on a single node with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 run_linear_eval.py \
    --model beit_base_patch16_224 \
    --pretrained_weights https://github.com/addf400/files/releases/download/v1.0/beit_base_patch16_224_pt22k.pth \
    --data_path /path/to/imagenet \
    --lr 4e-3 \
    --output_dir /path/to/save
```
