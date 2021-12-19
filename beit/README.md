# [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)

Official PyTorch implementation and pretrained models of BEiT.

- August 2021: [**BEiT**](https://huggingface.co/transformers/master/model_doc/beit.html) is on [HuggingFace](https://github.com/huggingface/transformers)
- July 2021: BEiT-large achieves **[state-of-the-art results on ADE20K](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k) (a big jump to 57.0 mIoU) for semantic segmentation**.
- July 2021: BEiT-large achieves **state-of-the-art ImageNet top-1 accuracy (88.6%) under the setting without extra data other than ImageNet-22k**.
- July 2021: release code and pretrained checkpoints (BEiT-base and BEiT-large)
- June 2021: release preprint in [arXiv](https://arxiv.org/abs/2106.08254)


---


## Pretrained models

We provide four BEiT weights pretrained on ImageNet-22k. The models were pretrained with 224x224 resolution.

- `BEiT-base`: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16 (#parameters: 86M)
- `BEiT-large`: #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16 (#parameters: 304M)

Download checkpoints that are **self-supervised pretrained and then intermediate fine-tuned** on ImageNet-22k (recommended):
- BEiT-base: [beit_base_patch16_224_pt22k_ft22k](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth)
- BEiT-large: [beit_large_patch16_224_pt22k_ft22k](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22k.pth)

Download checkpoints that are **self-supervised pretrained** on ImageNet-22k:
- BEiT-base: [beit_base_patch16_224_pt22k](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k.pth)
- BEiT-large: [beit_large_patch16_224_pt22k](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k.pth)


## Setup

```
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel bash
```

First, clone the repo and install required packages:
```
git clone https://github.com/microsoft/unilm.git
cd unilm/beit
pip install -r requirements.txt
```

The required packages including: [Pytorch](https://pytorch.org/) version 1.7.1, [torchvision](https://pytorch.org/vision/stable/index.html) version 0.8.2 and [Timm](https://github.com/rwightman/pytorch-image-models) version 0.3.2, etc.

For mixed-precision training, please install [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## Fine-tuning on ImageNet-1k (image classification)

We summarize the validation results as follows. We also provide the fine-tuned weights and fine-tuning logs. The detailed instructions to reproduce the results can be found at [`get_started_for_image_classification.md`](get_started_for_image_classification.md).

| name | initialized checkpoint | resolution | acc@1 | acc@5 | #params | weight | log |
|------------|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|-----|
| BEiT-base | [beit_base_patch16_224_pt22k](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k.pth) | 224x224 | 83.7 | 96.6 | 87M | [link](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft1k.pth) | [link](https://paste.ubuntu.com/p/79z5PncrKZ/) |
| BEiT-base | [beit_base_patch16_224_pt22k_ft22k](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth) | 224x224 | 85.2 | 97.6 | 87M | [link](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth) | [link](https://paste.ubuntu.com/p/KqFh55cwq4/) |
| BEiT-base | [beit_base_patch16_224_pt22k_ft22k](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth) | 384x384 | 86.8 | 98.1 | 87M | [link](https://unilm.blob.core.windows.net/beit/beit_base_patch16_384_pt22k_ft22kto1k.pth) | [link](https://paste.ubuntu.com/p/jnpD4NGZQn/) |
| BEiT-large | [beit_large_patch16_224_pt22k](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k.pth) | 224x224 | 86.0 | 97.6 | 304M | [link](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft1k.pth) | [link](https://paste.ubuntu.com/p/r4X4gHq6W5/) |
| BEiT-large | [beit_large_patch16_224_pt22k_ft22k](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 224x224 | 87.4 | 98.3 | 304M | [link](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22kto1k.pth) | [link](https://paste.ubuntu.com/p/DpHhW5Zgk5/) |
| BEiT-large | [beit_large_patch16_224_pt22k_ft22k](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 384x384 | 88.4 | 98.6 | 305M | [link](https://unilm.blob.core.windows.net/beit/beit_large_patch16_384_pt22k_ft22kto1k.pth) | [link](https://paste.ubuntu.com/p/xKTBDwPMd2/) |
| BEiT-large | [beit_large_patch16_224_pt22k_ft22k](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 512x512 | 88.60 | 98.66 | 306M | [link](https://unilm.blob.core.windows.net/beit/beit_large_patch16_512_pt22k_ft22kto1k.pth) | [link](https://paste.ubuntu.com/p/Wsb3NwkfCR/) |


## Fine-tuning on ADE20K (semantic segmentation)

We summarize the validation results as follows. We also provide the fine-tuned weights and fine-tuning logs. The detailed instructions to reproduce the results can be found at [`semantic_segmentation/README.md`](semantic_segmentation/README.md).

|name|initialized checkpoint|method|crop size|Lr schd|mIoU|mIoU (ms+flip)|#params|weight|log|
|:-----------|:---------------------|:-------:|:---------:|:-------:|:----:|:--------------:|:-------:|:-------|:---:|
|BEiT-base|[beit_base_patch16_224_pt22k_ft22k](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth)|UPerNet|640x640|160k|53.6|54.2|163M|[link](https://unilm.blob.core.windows.net/beit/beit_base_patch16_640_pt22k_ft22ktoade20k.pth)|[link](https://paste.ubuntu.com/p/FKc2cvvJsC/)|
|BEiT-large|[beit_large_patch16_224_pt22k_ft22k](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22k.pth)|UPerNet|640x640|160k|56.7|57.0|441M|[link](https://unilm.blob.core.windows.net/beit/beit_large_patch16_640_pt22k_ft22ktoade20k.pth)|[link](https://paste.ubuntu.com/p/sdsWCDRzk2/)|


## Example: Pre-training BEiT-base on ImageNet-22k

The BEiT-base model can be pretrained on ImageNet-22k using a DGX-2 box (16 V100-32GB):

```bash
# Set the path to save checkpoints
OUTPUT_DIR=/path/to/save/your_model
# Download and extract ImageNet-22k
DATA_PATH=/path/to/imagenet22k
# Download the tokenizer weight from OpenAI's DALL-E
TOKENIZER_PATH=/path/to/save/dall_e_tokenizer_weight
mkdir -p $TOKENIZER_PATH
wget -o $TOKENIZER_PATH/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl
wget -o $TOKENIZER_PATH/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=16 run_beit_pretraining.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --num_mask_patches 75 \
        --model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
        --batch_size 128 --lr 1.5e-3 --warmup_steps 10000 --epochs 150 \
        --clip_grad 3.0 --drop_path 0 --layer_scale_init_value 0.1
```
- `--num_mask_patches`: number of the input patches need be masked.
- `--batch_size`: batch size per GPU.
- Effective batch size = `number of GPUs` * `--batch_size`. So in the above example, the effective batch size is `128*16 = 2048`.
- `--lr`: learning rate.
- `--warmup_steps`: learning rate warmup steps.
- `--epochs`: total pre-training epochs.
- `--clip_grad`: clip gradient norm.
- `--drop_path`: stochastic depth rate.
- `--imagenet_default_mean_and_std`: enable this for ImageNet-1k pre-training, i.e., `(0.485, 0.456, 0.406)` for mean and `(0.229, 0.224, 0.225)` for std. We use `(0.5, 0.5, 0.5)` for mean and `(0.5, 0.5, 0.5)` for std by default on other pre-training data.
- `--layer_scale_init_value`: 0.1 for base, 1e-5 for large, set 0 to disable layerscale.

## Example: Pre-training BEiT-base on ImageNet-1k

The BEiT-base model can be pretrained on ImageNet-1k using a DGX-2 box (16 V100-32GB):
```bash
# Set the path to save checkpoints
OUTPUT_DIR=/path/to/save/your_model
# Download and extract ImageNet-1k
DATA_PATH=/path/to/imagenet1k_train_set
# Download the tokenizer weight from OpenAI's DALL-E
TOKENIZER_PATH=/path/to/save/dall_e_tokenizer_weight
mkdir -p $TOKENIZER_PATH
wget -o $TOKENIZER_PATH/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl
wget -o $TOKENIZER_PATH/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=16 run_beit_pretraining.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --num_mask_patches 75 \
        --model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
        --batch_size 128 --lr 1.5e-3 --warmup_epochs 10 --epochs 800 \
        --clip_grad 3.0 --drop_path 0 --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std
```

## Example: Fine-tuning BEiT on ImageNet-22k

The BEiT-large model can be fine-tuned on ImageNet-22k using a DGX-2 box (16 V100-32GB):

```bash
# Set the path to save checkpoints
OUTPUT_DIR=/path/to/save/your_model
# Download and extract ImageNet-22k
DATA_PATH=/path/to/imagenet22k

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=16 run_class_finetuning.py \
    --model beit_large_patch16_224 --data_path $DATA_PATH \
    --nb_classes 21841 --data_set image_folder --disable_eval_during_finetuning \
    --finetune https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k.pth \
    --output_dir $OUTPUT_DIR --batch_size 64 --lr 2e-3 --update_freq 2 \
    --warmup_epochs 5 --epochs 90 --layer_decay 0.75 --drop_path 0.2 \
    --weight_decay 0.05 --enable_deepspeed --layer_scale_init_value 1e-5 --clip_grad 1.0
```
- `--batch_size`: batch size per GPU.
- Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `16*64*2 = 2048`.
- `--lr`: learning rate.
- `--warmup_epochs`: learning rate warmup epochs.
- `--epochs`: total pre-training epochs.
- `--clip_grad`: clip gradient norm.
- `--drop_path`: stochastic depth rate.
- `--layer_scale_init_value`: 0.1 for base, 1e-5 for large, set 0 to disable layerscale.

The BEiT-base can be fine-tuned on ImageNet-22k as follows:

```bash
# Set the path to save checkpoints
OUTPUT_DIR=/path/to/save/your_model
# Download and extract ImageNet-22k
DATA_PATH=/path/to/imagenet22k

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=16 run_class_finetuning.py \
    --model beit_base_patch16_224 --data_path $DATA_PATH \
    --nb_classes 21841 --data_set image_folder --disable_eval_during_finetuning \
    --finetune https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k.pth \
    --output_dir $OUTPUT_DIR --batch_size 256 --lr 3e-3 --update_freq 1 \
    --warmup_epochs 5 --epochs 90 --layer_decay 0.65 --drop_path 0.2 \
    --weight_decay 0.05 --enable_deepspeed --layer_scale_init_value 0.1 --clip_grad 3.0
```

## Citation

If you find this repository useful, please consider citing our work:
```
@article{beit,
      title={{BEiT}: {BERT} Pre-Training of Image Transformers}, 
      author={Hangbo Bao and Li Dong and Furu Wei},
      year={2021},
      eprint={2106.08254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, the [DeiT](https://github.com/facebookresearch/deit) repository and the [Dino](https://github.com/facebookresearch/dino) repository.


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using BEiT models, please submit a GitHub issue.

For other communications related to UniLM AI, please contact Li Dong (`lidong1@microsoft.com`), [Furu Wei](http://gitnlp.org/) (`fuwei@microsoft.com`).
