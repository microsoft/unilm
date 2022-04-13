# DiT for Text Detection

This folder contains DiT running instructions on top of [Detectron2](https://github.com/facebookresearch/detectron2) for Text Detection.

<div align="center">
  <img src="https://user-images.githubusercontent.com/45008728/163219997-90d15c1b-e1d1-4bb3-ae46-774e54b89dc6.png" width="500" /><img src="https://user-images.githubusercontent.com/45008728/163220437-ab6a3fd2-0a4f-49c5-810c-e05dda7eb9e1.png" width="500"/> Text Detection outputs with FUNSD
</div>

## Usage

### Data Preparation

**FUNSD**

Follow [these steps](https://mmocr.readthedocs.io/en/latest/datasets/det.html#funsd) to download and process the dataset.

The resulting directory structure looks like the following:
```
│── data
│   ├── annotations
│   ├── imgs
│   ├── instances_test.json
│   └── instances_training.json

```

### Evaluation

The following commands provide examples to evaluate the fine-tuned checkpoint of DiT-Base with Mask R-CNN on the FUNSD.

The config files can be found in `configs`.

```bash
python train_net.py --config-file configs/mask_rcnn_dit_base.yaml --eval-only --num-gpus 8  --resume  MODEL.WEIGHTS path/to/model OUTPUT_DIR path/to/output
``` 

### Training
The following command provide example to train the Mask R-CNN with DiT backbone on 8 32GB Nvidia V100 GPUs.
```bash
python train_net.py --config-file configs/mask_rcnn_dit_base.yaml --num-gpus 8 --resume MODEL.WEIGHTS path/to/model OUTPUT_DIR path/to/output
``` 

## Acknowledgment
Thanks to [Detectron2](https://github.com/facebookresearch/detectron2) for Mask R-CNN implementation and [MMOCR](https://github.com/open-mmlab/mmocr) for the data preprocessing implementation of the FUNSD  
