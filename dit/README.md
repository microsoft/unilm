# [DiT: Self-Supervised Pre-Training for Document Image Transformer](https://arxiv.org/abs/2203.02378)

DiT (Document Image Transformer) is a self-supervised pre-trained Document Image Transformer model using large-scale unlabeled text images for Document AI tasks, which is essential since no supervised counterparts ever exist due to the lack of human labeled document images. 

<div align="center">
  <img src="https://user-images.githubusercontent.com/45008728/157173825-0949218a-61f5-4acb-949b-bbc499ab49f2.png" width="500" /><img src="https://user-images.githubusercontent.com/45008728/157173843-796dc878-2607-48d7-85cb-f54a2c007687.png" width="500"/> Model outputs with PubLayNet (left) and ICDAR 2019 cTDaR (right)
</div>

## What's New
- Demos on HuggingFace: [Document Layout Analysis](https://huggingface.co/spaces/nielsr/dit-document-layout-analysis), [Document Image Classification](https://huggingface.co/spaces/microsoft/document-image-transformer)
- March 2022: release pre-trained checkpoints and fine-tuning codes (DiT-base and DiT-large)
- March 2022: release preprint in [arXiv](https://arxiv.org/abs/2203.02378)

## Pretrained models

We provide two DiT weights pretrained on [IIT-CDIP Test Collection 1.0](https://dl.acm.org/doi/10.1145/1148170.1148307). The models were pretrained with 224x224 resolution.

- `DiT-base`: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16 (#parameters: 86M)
- `DiT-large`: #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16 (#parameters: 304M)

Download checkpoints that are **self-supervised pretrained** on IIT-CDIP Test Collection 1.0:
- DiT-base: [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D)
- DiT-large: [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D)

**If any file on this page fails to download, please add the following string as a suffix to the URL.**

**Suffix String:** ?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D

## Setup

First, clone the repo and install required packages:
```
git clone https://github.com/microsoft/unilm.git
cd unilm/dit
pip install -r requirements.txt
```

The required packages including: [Pytorch](https://pytorch.org/) version 1.9.0, [torchvision](https://pytorch.org/vision/stable/index.html) version 0.10.0 and [Timm](https://github.com/rwightman/pytorch-image-models) version 0.5.4, etc.

For mixed-precision training, please install [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
For object detection, please additionally install detectron2 library and shapely. Refer to the [Detectron2's INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

```bash
# Install `detectron2`
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# Install `shapely`
pip install shapely
```

## Fine-tuning on RVL-CDIP (Document Image Classification)

We summarize the validation results as follows. We also provide the fine-tuned weights. The detailed instructions to reproduce the results can be found at [`classification/README.md`](classification/README.md).

| name | initialized checkpoint | resolution | accuracy  | weight |
|------------|:----------------------------------------|:----------:|:-------:|-----|
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | 224x224 | 92.11 | [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/rvlcdip_dit-b.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |
| DiT-large | [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | 224x224 | 92.69 | [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/rvlcdip_dit-l.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |


## Fine-tuning on PubLayNet (Document Layout Analysis)

We summarize the validation results as follows. We also provide the fine-tuned weights. The detailed instructions to reproduce the results can be found at [`object_detection/README.md`](object_detection/README.md).

| name | initialized checkpoint | detection algorithm  |  mAP| weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Mask R-CNN | 0.935 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |
| DiT-large | [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Mask R-CNN | 0.941 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-l_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | 
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Cascade R-CNN | 0.945 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |
| DiT-large | [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Cascade R-CNN | 0.949 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-l_cascade.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |

## Fine-tuning on ICDAR 2019 cTDaR (Table Detection)

We summarize the validation results as follows. We also provide the fine-tuned weights. The detailed instructions to reproduce the results can be found at [`object_detection/README.md`](object_detection/README.md).

**Modern**

| name | initialized checkpoint | detection algorithm  |  Weighted Average F1 | weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Mask R-CNN | 94.74 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19modern_dit-b_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |
| DiT-large | [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Mask R-CNN | 95.50 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19modern_dit-l_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | 
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Cascade R-CNN | 95.85 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19modern_dit-b_cascade.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |
| DiT-large | [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Cascade R-CNN | 96.29 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19modern_dit-l_cascade.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |

**Archival**

| name | initialized checkpoint | detection algorithm  |  Weighted Average F1 | weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Mask R-CNN | 96.24 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19archival_dit-b_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |
| DiT-large | [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Mask R-CNN | 96.46 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19archival_dit-l_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | 
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Cascade R-CNN | 96.63 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19archival_dit-b_cascade.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |
| DiT-large | [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Cascade R-CNN | 97.00 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/icdar19archival_dit-l_cascade.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |

**Combined (Combine the inference results of Modern and Archival)**

| name | initialized checkpoint | detection algorithm  |  Weighted Average F1 | weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Mask R-CNN | 95.30 |  - |
| DiT-large | [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Mask R-CNN | 95.85 |  - | 
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Cascade R-CNN | 96.14 |  - |
| DiT-large | [dit_large_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-large-224-p16-500k-d7a2fb.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) | Cascade R-CNN | 96.55 |  - |


## Citation

If you find this repository useful, please consider citing our work:
```
@misc{li2022dit,
    title={DiT: Self-supervised Pre-training for Document Image Transformer},
    author={Junlong Li and Yiheng Xu and Tengchao Lv and Lei Cui and Cha Zhang and Furu Wei},
    year={2022},
    eprint={2203.02378},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, the [detectron2](https://github.com/facebookresearch/detectron2) library, the [DeiT](https://github.com/facebookresearch/deit) repository, the [Dino](https://github.com/facebookresearch/dino) repository, the [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository and the [MPViT](https://github.com/youngwanLEE/MPViT) repository.


### Contact Information

For help or issues using DiT models, please submit a GitHub issue.

For other communications related to DiT, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).
