# ADE20k Semantic Segmentation with BEiT

## Getting Started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

2. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md) to prepare the ADE20k dataset.


## Fine-tuning

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --seed 0  --deterministic --options model.pretrained=<IMAGENET_CHECKPOINT_PATH/URL>
```

Using a BEiT-base backbone with UperNet:
```bash
bash tools/dist_train.sh \
    configs/beit/upernet/upernet_beit_base_12_512_slide_160k_21ktoade20k.py 8 \
    --work-dir /path/to/save --seed 0  --deterministic \
    --options model.pretrained=https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth
```

Using a BEiT-large backbone with UperNet:
```bash
bash tools/dist_train.sh \
    configs/beit/upernet/upernet_beit_large_24_512_slide_160k_21ktoade20k.py 8 \
    --work-dir /path/to/save --seed 0  --deterministic \
    --options model.pretrained=https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth
```

## Evaluation

Command format:
```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate a BEiT-large backbone with UperNet:
```bash
bash tools/dist_test.sh configs/beit/upernet/upernet_beit_large_24_512_slide_160k_21ktoade20k.py \
    https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21ktoade20k.pth  4 --eval mIoU 
```

Expected results:
```
+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 57.54 | 68.78 | 86.22 |
+--------+-------+-------+-------+
```

---

## Acknowledgment 

This code is built using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, [Timm](https://github.com/rwightman/pytorch-image-models) library, the [Swin](https://github.com/microsoft/Swin-Transformer) repository, [XCiT](https://github.com/facebookresearch/xcit) and the [SETR](https://github.com/fudan-zvg/SETR) repository.
