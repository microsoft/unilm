# ADE20k Semantic segmentation with BEiT

## Getting started 

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

3. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.


## Fine-tuning

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --seed 0  --deterministic --options model.pretrained=<IMAGENET_CHECKPOINT_PATH/URL>
```

For example, using a BEiT-base backbone with UperNet:
```bash
bash tools/dist_train.sh \
    configs/beit/upernet/upernet_beit_base_12_640_slide_160k_ade20k_pt2ft.py 8 \
    --work-dir /path/to/save --seed 0  --deterministic \
    --options model.pretrained=https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth
```

More config files can be found at [`configs/beit/upernet`](configs/beit/upernet).


## Evaluation

Command format:
```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate a BEiT-base backbone with UperNet:
```bash
bash tools/dist_test.sh configs/beit/upernet/upernet_beit_base_12_640_slide_160k_ade20k_pt2ft.py \ 
    https://unilm.blob.core.windows.net/beit/beit_base_patch16_640_pt22k_ft22ktoade20k.pth  4 --eval mIoU
```

Expected results:
```
+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 53.61 | 64.82 | 84.62 |
+--------+-------+-------+-------+
```

Multi-scale + flip (`\*_ms.py`)
```
bash tools/dist_test.sh configs/beit/upernet/upernet_beit_base_12_640_slide_160k_ade20k_ms.py \
    https://unilm.blob.core.windows.net/beit/beit_base_patch16_640_pt22k_ft22ktoade20k.pth  4 --eval mIoU
```

Expected results:
```
+--------+-------+-------+------+
| Scope  | mIoU  | mAcc  | aAcc |
+--------+-------+-------+------+
| global | 54.26 | 65.28 | 84.9 |
+--------+-------+-------+------+
```

---

## Acknowledgment 

This code is built using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, [Timm](https://github.com/rwightman/pytorch-image-models) library, the [Swin](https://github.com/microsoft/Swin-Transformer) repository, [XCiT](https://github.com/facebookresearch/xcit) and the [SETR](https://github.com/fudan-zvg/SETR) repository.
