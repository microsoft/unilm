# VLMo - General-purpose Multimodal Pre-training

Paper: [VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/abs/2111.02358).

Official PyTorch implementation and pre-trained models of VLMo.

- Dec, 2022: Code & model release. 
- Sep, 2022: [**VLMo**](https://arxiv.org/pdf/2111.02358.pdf) was accepted by NeurIPS 2022.
- May 30th, 2022: new version of [**VLMo** paper on arXiv](https://arxiv.org/pdf/2111.02358.pdf).
- November 24th, 2021: **VLMo** Large (**single** model) as the new SOTA on the [VQA Challenge](https://eval.ai/web/challenges/challenge-page/830/leaderboard/2278)
- Nov 2021: release preprint in [arXiv](https://arxiv.org/abs/2111.02358)

## Pre-trained Models

We provide three VLMo weights pre-trained on COCO, VG, SBU and GCC. The models were pre-trained with 224x224 resolution.

- [`VLMo-base`](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_224.pt): #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16; #VL_FFN=2 (#parameters: 175M)
- [`VLMo-base_plus`](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_224.pt): #layer=24; hidden=544; FFN factor=4x; #head=16; patch=16x16; #VL_FFN=3 (#parameters: 167M)
- [`VLMo-large`](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_224.pt): #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16; #VL_FFN=3 (#parameters: 562M)

## Setup

```
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel bash
```

First, clone the repo and install required packages:
```
git clone https://github.com/microsoft/unilm.git
cd unilm/vlmo

pip install -r requirements.txt
```

## Dataset Preparation

We process the pre-training and fine-tuning data to the same format as in [ViLT](DATA.md).

## Pre-training

Replace `<ARROW_ROOT>` as your data dir in following commands.

### Step 1: Vision Pre-Training

Download the pre-trained model weight from [BEiT repo](https://github.com/microsoft/unilm/tree/master/beit).

### Step 2: Language Pre-Training (VLMo-Base)

```bash
# download from https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth
export INIT_CKPT=/path/to/save/beit_base_checkpoint

python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_textmlm_base whole_word_masking=True step200k per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=$INIT_CKPT log_dir=<YOUR_OUTPUT_PATH>
```
Or you can download our pre-trained ckpts for this stage:
- [`VLMo-base-stage2`](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_224_stage2.pt)
- [`VLMo-base_plus-stage2`](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_224_stage2.pt)
- [`VLMo-large-stage2`](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_224_stage2.pt)

### Step 3: Vision-Language Pre-Training (VLMo-Base)

```bash

export INIT_CKPT=/path/to/save/last_stage_ckpt

python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_mlm_itm_itc_base whole_word_masking=True step200k per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=$INIT_CKPT log_dir=<YOUR_OUTPUT_PATH>
```

## Fine-Tuning on Downstream Tasks

## Commands

```bash

python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> "<CONFIG_NAME>" per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path="<VLMo_WEIGHT>" log_dir=<YOUR_OUTPUT_PATH>
```

To reduce GPU memory cost, use [Deepspeed](https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed-zero-stage-1) and [Activation Checkpoint](https://fairscale.readthedocs.io/en/stable/api/nn/checkpoint/checkpoint_activations.html).

## Configs

You can found "<CONFIG_NAME>" for each task as follows:

### VQAv2
| <CONFIG_NAME> | initialized checkpoint | finetuned weight | test-dev |
|---------------|:----------------------:|:----------------:|:-----------:|
|task_finetune_vqa_base_image480|[VLMo-base](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_480_vqa.pt)|76.6|
|task_finetune_vqa_base_plus_image480|[VLMo-base_plus](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_480_vqa.pt)|78.5|
|task_finetune_vqa_large_image480|[VLMo-large](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_480_vqa.pt)|79.9|

### NLVR2
| <CONFIG_NAME> | initialized checkpoint | finetuned weight | test-P |
|---------------|:----------------------:|:----------------:|:-----------:|
|task_finetune_nlvr2_base_image384|[VLMo-base](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_384_nlvr2.pt)|83.3|
|task_finetune_nlvr2_base_plus_image384|[VLMo-base_plus](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_384_nlvr2.pt)|85.1|
|task_finetune_nlvr2_large_image384|[VLMo-large](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_384_nlvr2.pt)|86.9|

### COCO
| <CONFIG_NAME> | initialized checkpoint | finetuned weight | TR@1 | IR@1 |
|---------------|:----------------------:|:----------------:|:-----------:|:---:|
|task_finetune_irtr_coco_base_image384|[VLMo-base](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_384_coco.pt)|74.8|57.2|
|task_finetune_irtr_coco_base_plus_image384|[VLMo-base_plus](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_384_coco.pt)|76.3|58.6|
|task_finetune_irtr_coco_large_image384|[VLMo-large](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_384_coco.pt)|78.2|60.6|

### F30K
| <CONFIG_NAME> | initialized checkpoint | finetuned weight | TR@1 | IR@1 |
|---------------|:----------------------:|:----------------:|:-----------:|:---:|
|task_finetune_irtr_f30k_base_image384|[VLMo-base_coco_finetuned](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_384_coco.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_patch16_384_f30k.pt)|92.3|79.3|
|task_finetune_irtr_f30k_base_plus_image384|[VLMo-base_plus](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_224.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_base_plus_patch16_384_f30k.pt)|93.2|81.8|
|task_finetune_irtr_f30k_large_image384|[VLMo-large_coco_finetuned](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_384_coco.pt)|[weight](https://conversationhub.blob.core.windows.net/beit-share-public/vlmo/vlmo_large_patch16_384_f30k.pt)|95.3|84.5|

## Evaluation

To eval a finetuned model by appending `test_only=True` and set `load_path=` to the finetuned VLMo weight as follow:

```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=1 "<CONFIG_NAME>" per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path="<Finetuned_VLMo_WEIGHT>" test_only=True
```
- For retrieval tasks, also set `get_recall_metric=True` in the command.

## Acknowledgement

This repository is built using the [ViLT](https://github.com/dandelin/ViLT) repository, [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository, [ALBEF](https://github.com/salesforce/ALBEF) and the [timm](https://github.com/rwightman/pytorch-image-models) library.

## Citation

If you find this repository useful, please consider citing our work:
```
@inproceedings{vlmo,
      title={{VLMo}: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts},
      author={Hangbo Bao and Wenhui Wang and Li Dong and Qiang Liu and Owais Khan Mohammed and Kriti Aggarwal and Subhojit Som and Songhao Piao and Furu Wei},
      booktitle={Advances in Neural Information Processing Systems},
      year={2022},
      url={https://openreview.net/forum?id=bydKs84JEyw}
}
```


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using VLMo models, please submit a GitHub issue.
