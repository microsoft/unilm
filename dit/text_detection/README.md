# DiT for Text Detection

This folder contains DiT running instructions on top of [Detectron2](https://github.com/facebookresearch/detectron2) for Text Detection.

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
