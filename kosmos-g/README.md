# Kosmos-G: Generating Images in Context with Multimodal Large Language Models
[Paper](https://arxiv.org/abs/2310.02992) | [Project Page](https://xichenpan.github.io/kosmosg/)

## Checkpoints

Download checkpoints for stage1, stage2, and the final model.

```shell
mkdir kosmosg_checkpoints
cd kosmosg_checkpoints
DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL2tvc21vc2cvVmlULUwtMTQtc2QucHQ/c3Y9MjAyMy0wMS0wMyZzdD0yMDI0LTA0LTEwVDEzJTNBMTElM0E0NFomc2U9MjA1MC0wNC0xMVQxMyUzQTExJTNBMDBaJnNyPWMmc3A9ciZzaWc9NGNYSklqVlJaSElCV3FIalBnRG4lMkYwMW9jenBEV1hpcG1QQ1VrM1o4dmJRJTNE" | base64 --decode)
wget -O ViT-L-14-sd.pt $DLINK
DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL2tvc21vc2cvY2hlY2twb2ludF9zdGFnZTEucHQ/c3Y9MjAyMy0wMS0wMyZzdD0yMDI0LTA0LTEwVDEzJTNBMTElM0E0NFomc2U9MjA1MC0wNC0xMVQxMyUzQTExJTNBMDBaJnNyPWMmc3A9ciZzaWc9NGNYSklqVlJaSElCV3FIalBnRG4lMkYwMW9jenBEV1hpcG1QQ1VrM1o4dmJRJTNE" | base64 --decode)
wget -O checkpoint_stage1.pt $DLINK
DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL2tvc21vc2cvY2hlY2twb2ludF9zdGFnZTIucHQ/c3Y9MjAyMy0wMS0wMyZzdD0yMDI0LTA0LTEwVDEzJTNBMTElM0E0NFomc2U9MjA1MC0wNC0xMVQxMyUzQTExJTNBMDBaJnNyPWMmc3A9ciZzaWc9NGNYSklqVlJaSElCV3FIalBnRG4lMkYwMW9jenBEV1hpcG1QQ1VrM1o4dmJRJTNE" | base64 --decode)
wget -O checkpoint_stage2.pt $DLINK
DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL2tvc21vc2cvY2hlY2twb2ludF9maW5hbC5wdD9zdj0yMDIzLTAxLTAzJnN0PTIwMjQtMDQtMTBUMTMlM0ExMSUzQTQ0WiZzZT0yMDUwLTA0LTExVDEzJTNBMTElM0EwMFomc3I9YyZzcD1yJnNpZz00Y1hKSWpWUlpISUJXcUhqUGdEbiUyRjAxb2N6cERXWGlwbVBDVWszWjh2YlElM0Q=" | base64 --decode)
wget -O checkpoint_final.pt $DLINK
```

## Setup

### Using Docker Image [Recommended]

You can use our built Docker Image

```bash
docker run --runtime=nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name kosmosg --privileged=true -it -v /mnt:/mnt/ xichenpan/kosmosg:v1 /bin/bash
git clone https://github.com/microsoft/unilm.git
cd unilm/kosmos-g
pip install torchscale/
pip install open_clip/
pip install fairseq/
pip install infinibatch/
```

You can also start with NVIDIA Official Docker Image, and install all dependencies manually.

```bash
docker run --runtime=nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name kosmosg --privileged=true -it -v /mnt:/mnt/ nvcr.io/nvidia/pytorch:22.10-py3 /bin/bash
apt-get install -y libsm6 libxext6 libxrender-dev
git clone https://github.com/microsoft/unilm.git
cd unilm/kosmos-g
bash vl_setup.sh
```

### Using Base Environment
Make sure you have Pytorch 1.13.0 and nvcc 11.x installed.
```bash
git clone https://github.com/microsoft/unilm.git
cd unilm/kosmos-g
bash vl_setup.sh
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
bash scripts/remove_dreambench_multiimg.sh /path/to/dreambench/dreambooth/dataset
```

Specify the cfg in `sample_kosmosg_dreambench.py` and run the script to evaluate:
```shell
bash runeval_dreambench.sh
```

## Citation

If you find this repository useful, please consider citing our work:
```
@article{kosmos-g,
  title={{Kosmos-G}: Generating Images in Context with Multimodal Large Language Models},
  author={Xichen Pan and Li Dong and Shaohan Huang and Zhiliang Peng and Wenhu Chen and Furu Wei},
  journal={ArXiv},
  year={2023},
  volume={abs/2310.02992}
}
```

## Acknowledgement

This repository is built using [torchscale](https://github.com/microsoft/torchscale), [fairseq](https://github.com/facebookresearch/fairseq), [openclip](https://github.com/mlfoundations/open_clip). We thank the authors of [Nerfies](https://github.com/nerfies/nerfies.github.io) that kindly open sourced the template of the project page.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using models, please submit a GitHub issue.
