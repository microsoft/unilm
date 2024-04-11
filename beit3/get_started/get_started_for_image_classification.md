# Fine-tuning BEiT-3 on ImageNet-1k (Image Classification)


## Setup

1. [Setup environment](../README.md#setup).
2. Download and extract ImageNet-1k from http://image-net.org/.

The directory structure is the standard layout of torchvision's [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder). The training and validation data are expected to be in the `train/` folder and `val/` folder, respectively:

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

We then generate the index json files using the following command. [beit3.spm](https://github.com/addf400/files/releases/download/beit3/beit3.spm) is the sentencepiece model used for tokenizing texts.
```
from datasets import ImageNetDataset

ImageNetDataset.make_dataset_index(
    train_data_path = "/path/to/your_data/train",
    val_data_path = "/path/to/your_data/val",
    index_path = "/path/to/your_data"
)
```


## Example: Fine-tuning BEiT-3 on ImageNet-1k (Image Classification)

The BEiT-3 **base** model can be finetuned on ImageNet-1k using 8 V100-32GB:

```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_224 \
        --task imagenet \
        --batch_size 128 \
        --layer_decay 0.65 \
        --lr 7e-4 \
        --update_freq 1 \
        --epochs 50 \
        --warmup_epochs 5 \
        --drop_path 0.15 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --dist_eval \
        --mixup 0.8 \
        --cutmix 1.0 \
        --enable_deepspeed
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*128*1 = 1024`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models)
- `--enable_deepspeed`: optional. If you use apex, please enable deepspeed.


The BEiT-3 **large** model can be finetuned on ImageNet-1k using a DGX box (8 V100-32GB):

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_large_patch16_224 \
        --task imagenet \
        --batch_size 128 \
        --layer_decay 0.8 \
        --lr 2e-4 \
        --update_freq 1 \
        --epochs 50 \
        --warmup_epochs 5 \
        --drop_path 0.25 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --dist_eval \
        --mixup 0.8 \
        --cutmix 1.0 \
        --enable_deepspeed \
        --checkpoint_activations
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*128 = 1024`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models)
- `--enable_deepspeed`: optional. If you use apex, please enable deepspeed.
- `--checkpoint_activations`: using gradient checkpointing for saving GPU memory

## Example: Evaluate BEiT-3 Finetuned model on ImageNet-1k (Image Classification)

- Evaluate our fine-tuned BEiT3-base model on ImageNet val with a single GPU:
```bash       
python -m torch.distributed.launch --nproc_per_node=1 run_beit3_finetuning.py \
        --model beit3_base_patch16_224 \
        --task imagenet \
        --batch_size 128 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224_in1k.pth \
        --data_path /path/to/your_data \
        --eval \
        --dist_eval
```

Expected results:
```
* Acc@1 85.400 Acc@5 97.630
```

- Evaluate our fine-tuned BEiT3-large model on ImageNet val with a single GPU:
```bash       
python -m torch.distributed.launch --nproc_per_node=1 run_beit3_finetuning.py \
        --model beit3_large_patch16_224 \
        --task imagenet \
        --batch_size 128 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_224_in1k.pth \
        --data_path /path/to/your_data \
        --eval \
        --dist_eval
```

Expected results:
```
* Acc@1 87.580 Acc@5 98.326
```
