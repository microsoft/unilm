# DiT for Text Detection

<div align="center">
  <img src="https://user-images.githubusercontent.com/45008728/163219997-90d15c1b-e1d1-4bb3-ae46-774e54b89dc6.png" width="500" /><img src="https://user-images.githubusercontent.com/45008728/163220437-ab6a3fd2-0a4f-49c5-810c-e05dda7eb9e1.png" width="500"/> Model outputs with FUNSD
</div>

## Fine-tuned models on FUNSD
We summarize the validation results as follows. We also provide the fine-tuned weights. 

| name | initialized checkpoint | detection algorithm  |  F1 | weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| DiT-base-syn | [dit_base_patch16_224_syn](https://layoutlm.blob.core.windows.net/dit/dit-fts/td-syn_dit-b_mrcnn.pth) | Mask R-CNN | 94.25 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/funsd_dit-b_mrcnn.pth) |
| DiT-large-syn | [dit_large_patch16_224_syn](https://layoutlm.blob.core.windows.net/dit/dit-fts/td-syn_dit-l_mrcnn.pth) | Mask R-CNN | 94.29 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/funsd_dit-l_mrcnn.pth) | 


## Usage

### Data Preparation

Follow [these steps](https://mmocr.readthedocs.io/en/latest/datasets/det.html#funsd) to download and process the FUNSD. The resulting directory structure looks like the following:
```
│── data
│   ├── annotations
│   ├── imgs
│   ├── instances_test.json
│   └── instances_training.json

```
### Training
The following command provide example to train the Mask R-CNN with DiT backbone on 8 32GB Nvidia V100 GPUs.

The config files can be found in `configs`.

```bash
python train_net.py --config-file configs/mask_rcnn_dit_base.yaml --num-gpus 8 --resume MODEL.WEIGHTS path/to/model OUTPUT_DIR path/to/output
``` 

### Evaluation
The following commands provide examples to evaluate the fine-tuned checkpoint of DiT-Base with Mask R-CNN.

```bash
python train_net.py --config-file configs/mask_rcnn_dit_base.yaml --eval-only --num-gpus 8  --resume  MODEL.WEIGHTS path/to/model OUTPUT_DIR path/to/output
``` 


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


## Acknowledgment
Thanks to [Detectron2](https://github.com/facebookresearch/detectron2) for Mask R-CNN implementation and [MMOCR](https://github.com/open-mmlab/mmocr) for the data preprocessing implementation of the FUNSD  
