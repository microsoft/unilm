# Kosmos-G: Generating Images in Context with Multimodal Large Language Models

## Checkpoints

Comming Soon.

## Setup

1. Download the recommended docker image and launch it:
```bash
alias=`whoami | cut -d'.' -f2`; docker run --runtime=nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name kosmosg --privileged=true -p 8600:22 -it -v /mnt:/mnt ${alias}.azurecr.io/xformers/a6000:v1 /bin/bash
```
2. Clone the repo:
```bash
git clone https://github.com/microsoft/unilm.git
cd unilm/kosmos-g
```
3. Install the packages:
```bash
bash vl_setup_xl.sh
```

## Demo

If you would like to host a local Gradio demo, run the following command after [setup](#setup):
```bash
bash runapp.sh
```
Be sure to adjust the guidance scale if you find the default one leads to over-saturated images.

## Training

### Preparing dataset

Refer to [this guide](scripts/README.md) to prepare the dataset.

### Train script
After preparing the data, run the following command to train the model. Be sure to change the directories in the script to your own.
For the image decoder aligning stage:
```bash
bash runalign.sh
```
For the instruction tuning stage:
```bash
bash runtrain.sh
```

## Evaluation

### FID score on COCO (2014) val set

Download and unzip the COCO (2014) val set:
```shell
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip val2014.zip
```
Specify the cfg in `sample_kosmosg_coco.py` and run the script to evaluate:
```shell
bash runeval_coco.sh
```

### DINO score, CLIP-I score and CLIP-T score on DreamBench
Download DreamBench:
```shell
mkdir dreambench
cd dreambench
git clone https://github.com/google/dreambooth.git
```

We keep only one image for each entity as described in our paper.
```
bash scripts/remove_dreambench_multiimg.sh /path/to/dataset
```

Specify the cfg in `sample_kosmosg_dreambench.py` and run the script to evaluate:
```shell
bash runeval_dreambench.sh
```

## Citation

If you find this repository useful, please consider citing our work:
```
@article{kosmos-1,
  title={Language Is Not All You Need: Aligning Perception with Language Models},
  author={Shaohan Huang and Li Dong and Wenhui Wang and Yaru Hao and Saksham Singhal and Shuming Ma and Tengchao Lv and Lei Cui and Owais Khan Mohammed and Qiang Liu and Kriti Aggarwal and Zewen Chi and Johan Bjorck and Vishrav Chaudhary and Subhojit Som and Xia Song and Furu Wei},
  journal={ArXiv},
  year={2023},
  volume={abs/2302.14045}
}

@article{metalm,
  title={Language Models are General-Purpose Interfaces},
  author={Yaru Hao and Haoyu Song and Li Dong and Shaohan Huang and Zewen Chi and Wenhui Wang and Shuming Ma and Furu Wei},
  journal={ArXiv},
  year={2022},
  volume={abs/2206.06336}
}
```

## Acknowledgement

This repository is built using [torchscale](https://github.com/microsoft/torchscale), [fairseq](https://github.com/facebookresearch/fairseq), [openclip](https://github.com/mlfoundations/open_clip). We thank the authors of [Nerfies](https://github.com/nerfies/nerfies.github.io) that kindly open sourced the template of the project page.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using models, please submit a GitHub issue.
