# Kosmos-2: Grounding Multimodal Large Language Models to the World

- June 2023: 🔥 We release the **Kosmos-2: Grounding Multimodal Large Language Models to the World** paper. Checkout the [paper](https://arxiv.org/abs/2306.14824) and [online demo](https://aka.ms/kosmos-2-demo).
- Feb 2023: [Kosmos-1 (Language Is Not All You Need: Aligning Perception with Language Models)](https://arxiv.org/abs/2302.14045)
- June 2022: [MetaLM (Language Models are General-Purpose Interfaces)](https://arxiv.org/abs/2206.06336)


## Contents

- [Pretrained Models](#checkpoints)
- [Setup](#setup)
- [Launch Demo Locally](#demo)
- [GRIT: Large-Scale Training Corpus of Grounded Image-Text Pairs](#grit-large-scale-training-corpus-of-grounded-image-text-pairs)
- [Evaluation](#evaluation)

## Checkpoints

To be uploaded.

## Setup

1. Download recommended docker image and launch it:
```bash
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel bash
```
2. Clone the repo:
```bash
git clone https://github.com/microsoft/unilm.git
cd unilm/kosmos-2
```
3. Install the packages:
```bash
bash vl_setup_xl.sh
``` 

## Demo

We host a public demo at [link](https://aka.ms/kosmos-2-demo). If you would like to host a local Gradio demo, run the following command after [setup](#setup):
```bash
# install gradio
pip install httpcore==0.15.0 gradio==3.9.0

bash run_gradio.sh
``` 

## GRIT: Large-Scale Training Corpus of Grounded Image-Text Pairs

We introduce GRIT, a large-scale dataset of **Gr**ounded **I**mage-**T**ext pairs, which is created based on image-text pairs from a subset of COYO-700M and LAION-2B.
We construct a pipeline to extract and link text spans (i.e., noun phrases, and referring expressions) in the caption to their corresponding image regions.
More details can be found in the paper.

We will update the download link within this week.

## Evaluation

To be updated.

## Citation

If you find this repository useful, please consider citing our work:
```
@article{kosmos-2,
  title={Kosmos-2: Grounding Multimodal Large Language Models to the World},
  author={Zhiliang Peng and Wenhui Wang and Li Dong and Yaru Hao and Shaohan Huang and Shuming Ma and Furu Wei},
  journal={ArXiv},
  year={2023},
  volume={abs/2306}
}

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

This repository is built using [torchscale](https://github.com/microsoft/torchscale), [fairseq](https://github.com/facebookresearch/fairseq), [openclip](https://github.com/mlfoundations/open_clip).


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using models, please submit a GitHub issue.
