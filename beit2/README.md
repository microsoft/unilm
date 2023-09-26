# [BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/abs/2208.06366)
![](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/acc_compare.jpg?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)

Official PyTorch implementation and pretrained models of **BEiT v2**. 

The code and pretrained models of **BEiT** can be found at [here](https://github.com/microsoft/unilm/tree/master/beit).

The code and pretrained models of **BEiT-3** can be found at [here](https://github.com/microsoft/unilm/tree/master/beit3).

- March, 2023: release [the code and pretrained models of **BEiT-3**](https://github.com/microsoft/unilm/tree/master/beit3)
- March, 2023: [**BEiT-3**](https://arxiv.org/abs/2208.10442) was accepted by **CVPR 2023**.
- Sept 2022: release [the code and pretrained models of **BEiT v2**](https://github.com/microsoft/unilm/tree/master/beit2)
- Aug 2022: release preprint [Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442)
- Aug 2022: release preprint [BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/abs/2208.06366)
- June 2022: release preprint [VL-BEiT: Generative Vision-Language Pretraining](https://arxiv.org/abs/2206.01127)
- March, 2022: add [linear probe examples](https://github.com/microsoft/unilm/blob/master/beit/get_started_for_image_classification.md#example-linear-probe-on-imagenet)
- January, 2022: [**BEiT**](https://openreview.net/forum?id=p-BhZSz59o4) was accepted by **ICLR 2022 as Oral presentation** (54 out of 3391).
- August 2021: [**BEiT**](https://huggingface.co/transformers/master/model_doc/beit.html) is on [HuggingFace](https://github.com/huggingface/transformers)
- July 2021: BEiT-large achieves **[state-of-the-art results on ADE20K](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k) (a big jump to 57.0 mIoU) for semantic segmentation**.
- July 2021: BEiT-large achieves **state-of-the-art ImageNet top-1 accuracy (88.6%) under the setting without extra data other than ImageNet-22k**.
- July 2021: release [the code and pretrained models of **BEiT**](https://github.com/microsoft/unilm/tree/master/beit)
- June 2021: release preprint [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)

## Pretrained Models

We provide four BEiT weights pretrained on **ImageNet-1k**. The models were pretrained with 224x224 resolution.

- `BEiT-base`: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16 (#parameters: 86M)
- `BEiT-large`: #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16 (#parameters: 304M)

Download checkpoints that are self-supervised pretrained on ImageNet-1k and then intermediate finetuned on ImageNet-21k (**recommended**):
- BEiT-base: [beitv2_base_patch16_224_pt1k_ft21k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)
- BEiT-large: [beitv2_large_patch16_224_pt1k_ft21k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)

Download checkpoints that are self-supervised pretrained on ImageNet-1k:
- BEiT-base: [beitv2_base_patch16_224_pt1k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)
- BEiT-large: [beitv2_large_patch16_224_pt1k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)

## Setup

```
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel bash
```

First, clone the repo and install required packages:
```
git clone https://github.com/microsoft/unilm.git
cd unilm/beit2
pip install -r requirements.txt
```

The required packages including: [Pytorch](https://pytorch.org/) version 1.7.1, [torchvision](https://pytorch.org/vision/stable/index.html) version 0.8.2 and [Timm](https://github.com/rwightman/pytorch-image-models) version 0.4.12, etc.

For mixed-precision training, please install [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Fine-tuning on ImageNet-1k (Image Classification)

We summarize the validation results as follows. We also provide the fine-tuned weights. The detailed instructions to reproduce the results can be found at [get_started_for_image_classification.md](get_started_for_image_classification.md).

| name | initialized checkpoint | resolution | acc@1 | acc@5 | #params | weight | 
|------------|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| BEiTv2-base | [beitv2_base_patch16_224_pt1k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) | 224x224 | 85.5 | 97.5 | 86.5M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) |
| BEiTv2-base | [beitv2_base_patch16_224_pt1k_ft21k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) | 224x224 | 86.5 | 98.0 | 86.5M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21kto1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) |
| BEiTv2-large | [beitv2_base_patch16_224_pt1k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) | 224x224 | 87.3 | 98.2 | 304M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) |
| BEiTv2-large | [beitv2_base_patch16_224_pt1k_ft21k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) | 224x224 | 88.4 | 98.6 | 304M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21kto1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) |

## Fine-tuning on ADE20K (Semantic Segmentation)

We summarize the validation results as follows. We also provide the fine-tuned weights. The detailed instructions to reproduce the results can be found at [`semantic_segmentation/README.md`](semantic_segmentation/README.md).

|name|initialized checkpoint|method|crop size|iterations|mIoU|#params|weight|
|:-----------|:---------------------|:-------:|:---------:|:-------:|:----:|:--------------:|:-------:|
|BEiTv2-base|[beitv2_base_patch16_224_pt1k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)|UPerNet|512x512|160k|53.1| 163M|[link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ftade20k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)|
|BEiTv2-base|[beitv2_base_patch16_224_pt1k_ft21k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)|UPerNet|512x512|160k| 53.5| 163M|[link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21ktoade20k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)|
|BEiTv2-large|[beitv2_large_patch16_224_pt1k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)|UPerNet|512x512|160k|56.7| 441M|[link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ftade20k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)|
|BEiTv2-large|[beitv2_large_patch16_224_pt1k_ft21k](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)|UPerNet|512x512|160k| 57.5| 441M|[link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21ktoade20k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)|


## Fine-tuning on MSCOCO2017 (Object Detection)
Under preparation.


## Pre-training on ImageNet-1k

See [PRETRAINING.md](PRETRAINING.md) for detailed instructions.


## Visual Tokenizer (VQ-KD) Trained on ImageNet-1k

We provide the VQ-KD tokenizer trained on **ImageNet-1k**.
- [vqkd_encoder_base_decoder_3x768x12](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D): #encoder layer=12; #decoder layer=3; hidden=768; FFN factor=4x; #head=12; patch=16x16;

See [TOKENIZER.md](TOKENIZER.md) for more details.

## Code for Analysis of Self-Attention Map

Pre-trained [BEiT_base_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt1k_800ep.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) on ImageNet-1k with 800 epochs, config: ``--disable_rel_pos_bias --abs_pos_emb --layer_scale_init_value 0``

```bash
python visualize_attention.py \
  --model beit_base_patch16_224_8k_vocab \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --layer_scale_init_value 0 \
  --input_size 480 \
  --pretrained_weights /folder/to/download/beit_base_patch16_224_pt1k_800ep.pth \
  --image_path ../visualization/input2.png \
  --selected_row 11 \
  --selected_col 13
```

`--selected_row 11` and `--selected_col` for choosing the image patch as query

## Citation

If you find this repository useful, please consider citing our work:
```
@inproceedings{beit,
title={{BEiT}: {BERT} Pre-Training of Image Transformers},
author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=p-BhZSz59o4}
}

@article{beitv2,
title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
year={2022},
eprint={2208.06366},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```


## Acknowledgement

This repository is built using the [BEiT](https://github.com/microsoft/unilm/tree/master/beit), the [CLIP](https://github.com/openai/CLIP), the [DeiT](https://github.com/facebookresearch/deit), the [Dino](https://github.com/facebookresearch/dino) repository and the [timm](https://github.com/rwightman/pytorch-image-models) library.


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using BEiT v2 models, please submit a GitHub issue.

For other communications, please contact Li Dong (`lidong1@microsoft.com`), [Furu Wei](http://gitnlp.org/) (`fuwei@microsoft.com`).
