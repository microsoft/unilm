# BEiT v2 Fine-tuning on ImageNet-1k (Image Classification)

## Model Zoo

We provide some finetuned models here.

| model name | pre-training epochs on ImageNet-1k | intermeidate fine-tuning epochs on ImageNet-21k | fine-tuning epochs on ImageNet-1k | weight | top-1 accuracy (%) |
|------------|:------------------:|:------:|:------:| :------:| :------:|
| beit_base_patch16_224 | 300 | 0 | 100 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_300e_ft1k.pth) | 85.0 |
| beit_base_patch16_224 | 1600 |0 | 100 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft1k.pth) | 85.5 |
| beit_base_patch16_224 | 1600 | 90 | 30 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21kto1k.pth) | 86.5 |
| beit_large_patch16_224 | 300 | 0 | 50 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_300e_ft1k.pth) | 86.6 |
| beit_large_patch16_224 | 1600 |0 | 50 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft1k.pth) | 87.3 |
| beit_large_patch16_224 | 1600 | 90 | 20 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21kto1k.pth) | 88.4 |


## Example: Fine-tuning BEiT v2 on ImageNet-1k (Image Classification)

The BEiT v2 **base** model can be finetuned on ImageNet-1k using a DGX box (8 V100-32GB):

```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
        --data_path /path/to/imagenet-1k/train \
        --eval_data_path /path/to/imagenet-1k/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model \
        --model beit_base_patch16_224 \
        --weight_decay 0.05 \
        --finetune https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth \
        --batch_size 128 \
        --lr 5e-5 \
        --update_freq 1 \
        --warmup_epochs 20 \
        --epochs 30 \
        --layer_decay 0.75 \
        --drop_path 0.1 \
        --mixup 0. \
        --cutmix 0. \
        --imagenet_default_mean_and_std \
        --dist_eval \
        --save_ckpt_freq 20 \
        --enable_deepspeed 
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*128*1 = 1024`.
- `--finetune`: weight path of your pretrained models; you can pretrain it by yourself, or download the pretrained model weights in [PRETRAINING.md](PRETRAINING.md)
- `--lr`: learning rate. 5e-4 for pretrained models and 5e-5 for intermediate fine-tuned models.
- `--epochs`: fine-tuning epochs. 100 for pretrained models and 30 for intermediate fine-tuned models.
- `--mixup`: 0.8 for pretrained models and 0. for intermediate fine-tuned models.
- `--cutmix`: 1.0 for pretrained models and 0. for intermediate fine-tuned models.
- `--layer_decay`: 0.6 for 1600 epochs pretrained models and 0.65 for 300 epochs. 0.75 for intermediate fine-tuned models.
- `--drop_path`: 0.2 for 1600 epochs pretrained models and 0.1 for 300 epochs. 0.1 for intermediate fine-tuned models.
- `--enable_deepspeed`: optional.


The BEiT v2 **large** model can be finetuned on ImageNet-1k using a DGX box (8 V100-32GB):

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
        --data_path /path/to/imagenet-1k/train \
        --eval_data_path /path/to/imagenet-1k/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model \
        --model beit_large_patch16_224 \
        --weight_decay 0.05 \
        --finetune https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth \
        --batch_size 64 \
        --lr 7e-5 \
        --update_freq 2 \
        --warmup_epochs 5 \
        --epochs 20 \
        --layer_decay 0.8 \
        --drop_path 0.25 \
        --mixup 0. \
        --cutmix 0. \
        --imagenet_default_mean_and_std   \
        --dist_eval \
        --save_ckpt_freq 10 \
        --enable_deepspeed 
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*64*2 = 1024`.
- `--finetune`: weight path of your pretrained models; you can pretrain it by yourself, or download the pretrained model weights in [PRETRAINING.md](PRETRAINING.md).
- `--lr`: learning rate. 2e-4 for 1600 epochs pretrained model and 5e-4 for 300 epochs. 7e-5 for intermediate fine-tuned models.
- `--epochs`: fine-tuning epochs. 50 for pretrained models and 20 for intermediate fine-tuned models.
- `--mixup`: 0.8 for pretrained models and 0. for intermediate fine-tuned models.
- `--cutmix`: 1.0 for pretrained models and 0. for intermediate fine-tuned models.
- `--layer_decay`: 0.8 for all models.
- `--drop_path`: 0.2 for pretrained models and 0.25 for intermediate fine-tuned models.
- `--enable_deepspeed`: optional.

## Example: Evaluate BEiT v2 Finetuned model on ImageNet-1k (Image Classification)

- Evaluate our fine-tuned BEiT-base model on ImageNet val with a single GPU:
```bash       
python -m torch.distributed.launch --nproc_per_node=1 run_class_finetuning.py \
        --data_path /path/to/imagenet-1k/train \
        --eval_data_path /path/to/imagenet-1k/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model beit_base_patch16_224 \
        --finetune https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21kto1k.pth \
        --batch_size 128 \
        --imagenet_default_mean_and_std \
        --dist_eval \
        --eval
```

Expected results:
```
* Acc@1 86.458 Acc@5 97.978 loss 0.569
```

- Evaluate our fine-tuned BEiT-large model on ImageNet val with a single GPU:
```bash       
python -m torch.distributed.launch --nproc_per_node=1 run_class_finetuning.py \
        --data_path /path/to/imagenet-1k/train \
        --eval_data_path /path/to/imagenet-1k/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model beit_large_patch16_224 \
        --finetune https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21kto1k.pth \
        --batch_size 128 \
        --imagenet_default_mean_and_std \
        --dist_eval \
        --eval
```

Expected results:
```
* Acc@1 88.370 Acc@5 98.578 loss 0.493
```

## Robust Evaluation on ImageNet Variants (Image Classification)

Download the datasets (ImageNet-Adversarial, ImageNet-Rendition, and ImageNet-Sketch) following [timm](https://github.com/rwightman/pytorch-image-models/blob/master/results/README.md).

For ImageNet-Rendition variants, one can test it like:
```bash       
python -m torch.distributed.launch --nproc_per_node=1 run_class_finetuning.py \
        --robust_test 'imagenet_r' \
        --data_path /path/to/imagenet-r \
        --eval_data_path /path/to/imagenet-r \
        --nb_classes 200 \
        --data_set image_folder \
        --model beit_large_patch16_224 \
        --finetune https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft1k.pth \
        --batch_size 128 \
        --imagenet_default_mean_and_std \
        --dist_eval \
        --save_ckpt_freq 20 \
        --eval
```
- `--robust_test`: `imagenet_r` for ImageNet-Rendition and `imagenet_a` for ImageNet-Adversarial.
- `--nb_classes`: 200 for ImageNet-Rendition and ImageNet-Adversarial, and 1000 for ImageNet-Sketch.


Expected results:
```
* Acc@1 69.940 Acc@5 82.890 loss 1.541
```


## Example: Intermediate fine-tuning BEiT v2 on ImageNet-21k
The BEiT v2 **base/large** model can be intermediate fine-tuned on ImageNet-21k using 2 DGX boxes (16 V100-32GB):
```bash
python -m torch.distributed.launch --nnodes 2 --node_rank {0, 1} --nproc_per_node=16  run_class_finetuning.py \
        --data_path /path/to/imagenet-21k/train \
        --disable_eval_during_finetuning \
        --nb_classes 21841 \
        --data_set image_folder \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model \
        --model beit_base_patch16_224 \
        --weight_decay 0.05 \
        --finetune /path/to/save/your_pretraining_model \
        --batch_size 128 \
        --lr 6e-4 \
        --update_freq 1 \
        --warmup_epochs 20 \
        --epochs 90 \
        --layer_decay 0.75 \
        --drop_path 0.1 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --imagenet_default_mean_and_std \
        --save_ckpt_freq 20 \
        --enable_deepspeed 
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `32*128*1 = 4096`.
- `--finetune`: weight path of your pretrained model; you can pretrain it by yourself, or download the pretrained model weight in [PRETRAINING.md](PRETRAINING.md)
- `--drop_path`: 0.1 for base model and 0.2 for large model.
- `--enable_deepspeed`: optional.

We provide some intermediate fine-tuned models here.
| model name | pre-training epochs on ImageNet-1k | intermediate fine-tuning epochs on ImageNet-21k | weight |
|------------|:------------------:|:------:|:------:|
| beit_base_patch16_224 | 1600 | 90 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth) |
| beit_large_patch16_224 | 1600 | 90 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth) |
