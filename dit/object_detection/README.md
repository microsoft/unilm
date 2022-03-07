# DiT for Object Detection

This folder contains Mask R-CNN Cascade Mask R-CNN running instructions on top of [Detectron2](https://github.com/facebookresearch/detectron2) for PubLayNet and ICDAR 2019 cTDaR.

## Usage
### Data Preparation

**PubLayNet**

Download the data from this [link](https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz?_ga=2.218138265.1825957955.1646384196-1495010506.1633610665) (~96GB). Then extract it to `PATH-to-PubLayNet`.

A soft link needs to be created to make the data accessible for the program:`ln -s PATH-to-PubLayNet publaynet_data`.

**ICDAR 2019 cTDaR**

Download the data from this [link](https://github.com/cndplab-founder/ICDAR2019_cTDaR) (~4GB). Assume path to this repository is named as `PATH-to-ICDARrepo`.

Then run `python convert_to_coco_format.py --root_dir=PATH-to-ICDARrepo --target_dir=PATH-toICDAR`. Now the path to processed data is `PATH-to-ICDAR`.

Run the following command to get the adaptively binarized images for archival subset.

```
cp -r PATH-to-ICDAR/trackA_archival PATH-to-ICDAR/at_trackA_archival
python adaptive_binarize.py --root_dir PATH-to-ICDAR/at_trackA_archival
```

The binarized archival subset will be in `PATH-to-ICDAR/at_trackA_archival`.

According to the subset you want to evaluate/fine-tune, a soft link should be created:`ln -s PATH-to-ICDAR/trackA_modern data` or `ln -s PATH-to-ICDAR/at_trackA_archival data`.

### Evaluation

Following commands provide two examples to evaluate the fine-tuned checkpoints.

The config files can be found in `icdar19_configs` and `publaynet_configs`.

1) Evaluate the fine-tuned checkpoint of DiT-Base with Mask R-CNN on PublayNet:
```bash
python train_net.py --config-file publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml --eval-only --num-gpus 8 MODEL.WEIGHTS <finetuned_checkpoint_file_path or link> OUTPUT_DIR <your_output_dir> 
``` 

2) Evaluate the fine-tuned checkpoint of DiT-Large with Cascade Mask R-CNN on ICDAR 2019 cTDaR archival subset (make sure you have created a soft link from `PATH-to-ICDAR/at_trackA_archival` to `data`):
```bash
python train_net.py --config-file icdar19_configs/cascade/cascade_dit_large.yaml --eval-only --num-gpus 8 MODEL.WEIGHTS <finetuned_checkpoint_file_path or link> OUTPUT_DIR <your_output_dir> 
```


### Training
The following commands provide two examples to train the Mask R-CNN/Cascade Mask R-CNN with DiT backbone on 8 32GB Nvidia V100 GPUs.

1) Fine-tune DiT-Base with Cascade Mask R-CNN on PublayNet:
```bash
python train_net.py --config-file publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 8 MODEL.WEIGHTS <DiT-Base_file_path or link> OUTPUT_DIR <your_output_dir> 
``` 


2) Fine-tune DiT-Large with Mask R-CNN on ICDAR 2019 cTDaR modern:
```bash
python train_net.py --config-file icdar19_configs/markrcnn/maskrcnn_dit_large.yaml --num-gpus 8 MODEL.WEIGHTS <DiT-Large_file_path or link> OUTPUT_DIR <your_output_dir> 
``` 

[Detectron2's document](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) may help you for more details.


## Acknowledgment
Thanks to [Detectron2](https://github.com/facebookresearch/detectron2) for Mask R-CNN and Cascade Mask R-CNN implementation.
