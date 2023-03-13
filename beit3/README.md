# [(BEiT-3) Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442)

Official PyTorch implementation and pretrained models of BEiT-3. 

The code and pretrained models of **BEiT** can be found at [here](https://github.com/microsoft/unilm/tree/master/beit).

The code and pretrained models of **BEiT v2** can be found at [here](https://github.com/microsoft/unilm/tree/master/beit2).

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

## Pretrained models

We provide BEiT-3 weights pretrained on monomodal and multimodal data. Our large-size model outperforms previous large-size models across various vision-language and vision downstream tasks. The models were pretrained with 224x224 resolution.

### Tips
- For vision-language tasks that require deep fusion, we recommend using `BEiT3-base` and `BEiT3-large`.
- For image-text retrieval or vision tasks, using `BEiT3-base-itc` and `BEiT3-large-itc` usually achieve better performance.

### Download Checkpoints

1. Models pretrained on ImageNet-21k images, 160 GB text documents, and web-scale image-text pairs (collected from [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/), [English LAION-2B](https://laion.ai/blog/laion-5b/), [COYO-700M](https://github.com/kakaobrain/coyo-dataset), and CC15M). 
   - [`BEiT3-base`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth): #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16; #parameters: 276M
   - [`BEiT3-large`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth): #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16; #parameters: 746M

2. Perform image-text contrastive intermediate tuning on `BEiT3-base` and `BEiT3-large`. 
   - [`BEiT3-base-itc`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_itc_patch16_224.pth): #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16; #parameters: 222M
   - [`BEiT3-large-itc`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_itc_patch16_224.pth): #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16; #parameters: 674M

3. Add indomain image-text pairs (COCO and VG) to continue training `BEiT3-base` and `BEiT3-large` using masked data modeling. The indomain models achieve better performance on VQAv2 and NLVR2 tasks.
   - [`BEiT3-base-indomain`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_indomain_patch16_224.pth): #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16; #parameters: 276M
   - [`BEiT3-large-indomain`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_indomain_patch16_224.pth): #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16; #parameters: 746M

### Architecture

We use [Magneto](https://arxiv.org/abs/2210.06423) with decoupled Multiway Transformer as the backbone architecture. Magneto can have better training stability and obtain better performance across modalities (such as vision, and language). The implementation is based on the [torchscale](https://github.com/microsoft/torchscale/blob/main/torchscale/model/BEiT3.py) package.


## Setup

```
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel bash
```

Clone the repo and install required packages:
```
git clone https://github.com/microsoft/unilm.git
cd unilm/beit3
pip install -r requirements.txt
```


## Fine-tuning on ImageNet-1k (Image Classification)

The detailed instructions can be found at [`get_started_for_image_classification.md`](get_started/get_started_for_image_classification.md). We only use vision-related parameters for image classification fine-tuning.

| initialized checkpoint | resolution | acc@1 | acc@5 | #params | weight |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [beit3_base_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth) | 224x224 | 85.4 | 97.6 | 87M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/in1k/beit3_base_patch16_224_in1k.pth) |
| [beit3_base_indomain_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_indomain_patch16_224.pth) | 224x224 | 85.4 | 97.6 | 87M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/in1k/beit3_base_indomain_patch16_224_in1k.pth) |
| [beit3_large_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth) | 224x224 | 87.6 | 98.3 | 305M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/in1k/beit3_large_patch16_224_in1k.pth) |
| [beit3_large_indomain_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_indomain_patch16_224.pth) | 224x224 | 87.5 | 98.3 | 305M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/in1k/beit3_large_indomain_patch16_224_in1k.pth) |


## Fine-tuning on VQAv2 (Visual Question Answering)

The detailed instructions can be found at [`get_started_for_vqav2.md`](get_started/get_started_for_vqav2.md).

| initialized checkpoint | resolution | augmented data | test-dev | test-std | #params | weight |
|:----------------------------------------|:----------:|:-----:|:-----:|:-----:|:-------:|-------------------|
| [beit3_base_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth) | 480x480 | - | 77.65 | - | 228M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/vqa/beit3_base_patch16_480_vqa.pth) |
| [beit3_base_indomain_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_indomain_patch16_224.pth) | 480x480 | - | 78.46 | - | 228M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/vqa/beit3_base_indomain_patch16_480_vqa.pth) |
| [beit3_large_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth) | 480x480 | - | 81.85 | - | 683M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/vqa/beit3_large_patch16_480_vqa.pth) |
| [beit3_large_indomain_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_indomain_patch16_224.pth) | 480x480 | - | 82.53 | - | 683M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/vqa/beit3_large_indomain_patch16_480_vqa.pth) |
| [beit3_large_indomain_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_indomain_patch16_224.pth) | 768x768 | VGQA | 82.97 | 83.03 | 684M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/vqa/beit3_large_indomain_patch16_768_vgqaaug_vqa.pth) |


## Fine-tuning on NLVR2 (Visual Reasoning)

The detailed instructions can be found at [`get_started_for_nlvr2.md`](get_started/get_started_for_nlvr2.md).

| initialized checkpoint | resolution | dev | test-P | #params | weight |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [beit3_base_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth) | 224x224 | 83.6 | 84.4 | 226M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/nlvr2/beit3_base_patch16_224_nlvr2.pth) |
| [beit3_base_indomain_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_indomain_patch16_224.pth) | 224x224 | 84.6 | 85.3 | 226M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/nlvr2/beit3_base_indomain_patch16_224_nlvr2.pth) |
| [beit3_large_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth) | 224x224 | 88.5 | 89.4 | 681M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/nlvr2/beit3_large_patch16_224_nlvr2.pth) |
| [beit3_large_indomain_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_indomain_patch16_224.pth) | 224x224 | 89.2 | 90.0 | 681M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/nlvr2/beit3_large_indomain_patch16_224_nlvr2.pth) |


## Fine-tuning on COCO Captioning and NoCaps (Image Captioning)

The detailed instructions can be found at [`get_started_for_image_captioning.md`](get_started/get_started_for_captioning.md).

### COCO Captioning

| initialized checkpoint | resolution | test CIDEr | #params | weight |
|:----------------------------------------|:----------:|:-----:|:-------:|-------------------|
| [beit3_base_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth) | 480x480 | 133.6 | 271M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/coco_captioning/beit3_base_patch16_480_coco_captioning.pth) |
| [beit3_base_indomain_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_indomain_patch16_224.pth) | 480x480 | 135.0 | 271M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/coco_captioning/beit3_base_indomain_patch16_480_coco_captioning.pth) |
| [beit3_large_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth) | 480x480 | 143.2 | 739M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/coco_captioning/beit3_large_patch16_480_coco_captioning.pth) |

### NoCaps

| initialized checkpoint | resolution | val CIDEr | #params | weight |
|:----------------------------------------|:----------:|:-----:|:-------:|-------------------|
| [beit3_base_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth) | 480x480 | 104.4 | 271M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/nocaps/beit3_base_patch16_480_nocaps.pth) |
| [beit3_base_indomain_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_indomain_patch16_224.pth) | 480x480 | 105.6 | 271M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/nocaps/beit3_base_indomain_patch16_480_nocaps.pth) |
| [beit3_large_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth) | 480x480 | 120.2 | 739M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/nocaps/beit3_large_patch16_480_nocaps.pth) |


## Fine-tuning on COCO and Flickr30k Retrieval (Image-Text Retrieval)

The detailed instructions can be found at [`get_started_for_retrieval.md`](get_started/get_started_for_retrieval.md).

### COCO Retrieval

| initialized checkpoint | resolution | IR@1 | TR@1 | #params | weight |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [beit3_base_itc_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_itc_patch16_224.pth) | 384x384 | 61.4 | 79.1 | 222M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/coco_retrieval/beit3_base_patch16_384_coco_retrieval.pth) |
| [beit3_large_itc_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_itc_patch16_224.pth) | 384x384 | 63.4 | 82.1 | 675M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/coco_retrieval/beit3_large_patch16_384_coco_retrieval.pth) |

### Flickr30k Retrieval

| initialized checkpoint | resolution | IR@1 | TR@1 | #params | weight |
|:----------------------------------------|:----------:|:-----:|:-----:|:-------:|-------------------|
| [beit3_base_itc_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_itc_patch16_224.pth) | 384x384 | 86.2 | 96.3 | 222M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/f30k_retrieval/beit3_base_patch16_384_f30k_retrieval.pth) |
| [beit3_large_itc_patch16_224](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_itc_patch16_224.pth) | 384x384 | 88.1 | 97.2 | 675M | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/f30k_retrieval/beit3_large_patch16_384_f30k_retrieval.pth) |


## Citation

If you find this repository useful, please consider citing our work:
```
@inproceedings{beit3,
title={Image as a foreign language: {BEiT} pretraining for vision and vision-language tasks},
author={Wenhui Wang and Hangbo Bao and Li Dong and Johan Bjorck and Zhiliang Peng and Qiang Liu and Kriti Aggarwal and Owais Khan Mohammed and Saksham Singhal and Subhojit Som and Furu Wei},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2023}
}

@article{beitv2,
title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
year={2022},
eprint={2208.06366},
archivePrefix={arXiv},
primaryClass={cs.CV}
}

@inproceedings{beit,
title={{BEiT}: {BERT} Pre-Training of Image Transformers},
author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=p-BhZSz59o4}
}
```


## Acknowledgement

This repository is built using the [BEiT](https://github.com/microsoft/unilm/tree/master/beit), the [BEiTv2](https://github.com/microsoft/unilm/tree/master/beit2), the [CLIP](https://github.com/openai/CLIP), the [open_clip](https://github.com/mlfoundations/open_clip), the [Oscar](https://github.com/microsoft/Oscar), the [DeiT](https://github.com/facebookresearch/deit), the [Dino](https://github.com/facebookresearch/dino) repository and the [timm](https://github.com/rwightman/pytorch-image-models) library.


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using BEiT-3 models, please submit a GitHub issue.
