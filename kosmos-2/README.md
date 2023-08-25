# Kosmos-2: Grounding Multimodal Large Language Models to the World
[[paper]](https://arxiv.org/abs/2306.14824) [[dataset]](https://huggingface.co/datasets/zzliang/GRIT)
[[online demo hosted by HuggingFace]](https://huggingface.co/spaces/ydshieh/Kosmos-2)

- Aug 2023: We acknowledge [ydshieh](https://huggingface.co/ydshieh) at HuggingFace for [the online demo](https://huggingface.co/spaces/ydshieh/Kosmos-2) and [the HuggingFace's transformers implementation](https://huggingface.co/ydshieh/kosmos-2-patch14-224).
- June 2023: 🔥 We release the **Kosmos-2: Grounding Multimodal Large Language Models to the World** paper. Checkout the [paper](https://arxiv.org/abs/2306.14824).
- Feb 2023: [Kosmos-1 (Language Is Not All You Need: Aligning Perception with Language Models)](https://arxiv.org/abs/2302.14045)
- June 2022: [MetaLM (Language Models are General-Purpose Interfaces)](https://arxiv.org/abs/2206.06336)


## Contents

- [Kosmos-2: Grounding Multimodal Large Language Models to the World](#kosmos-2-grounding-multimodal-large-language-models-to-the-world)
  - [Contents](#contents)
  - [Checkpoints](#checkpoints)
  - [Setup](#setup)
  - [Demo](#demo)
  - [GRIT: Large-Scale Training Corpus of Grounded Image-Text Pairs](#grit-large-scale-training-corpus-of-grounded-image-text-pairs)
    - [Download Data](#download-data)
  - [Evaluation](#evaluation)
    - [1. Phrase grounding](#1-phrase-grounding)
    - [2. Referring expression comprehension](#2-referring-expression-comprehension)
    - [3. Referring expression generation](#3-referring-expression-generation)
    - [4. Image captioning](#4-image-captioning)
    - [5. Visual question answering](#5-visual-question-answering)
  - [Training](#training)
    - [1. Preparing data](#preparing-dataset)
    - [2. Training script](#train-script)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)
  - [License](#license)
    - [Contact Information](#contact-information)

## Checkpoints

The model can be loaded with [the HuggingFace's transformers library](https://huggingface.co/ydshieh/kosmos-2-patch14-224).

The checkpoint can be downloaded from [here](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/kosmos-2.pt?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D):
```bash
wget -O kosmos-2.pt "https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/kosmos-2.pt?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D"
```

## Setup

1. Download the recommended docker image and launch it:
```bash
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} nvcr.io/nvidia/pytorch:22.10-py3 bash
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
(Refer to [comment](https://github.com/microsoft/unilm/issues/1204#issuecomment-1639812388) for detailed package info)

Alternatively, you can refer to [this guide](docs/install.md) to set up a conda environment.

## Demo

We acknowledge [ydshieh](https://huggingface.co/ydshieh) at HuggingFace for implementing [an online demo](https://huggingface.co/spaces/ydshieh/Kosmos-2).

<!-- We host a public demo at [link](https://aka.ms/kosmos-2-demo). -->
If you would like to host a local Gradio demo, run the following command after [setup](#setup):
```bash
bash run_gradio.sh
``` 

## GRIT: Large-Scale Training Corpus of Grounded Image-Text Pairs

We introduce GRIT, a large-scale dataset of **Gr**ounded **I**mage-**T**ext pairs, which is created based on image-text pairs from a subset of COYO-700M and LAION-2B.
We construct a pipeline to extract and link text spans (i.e., noun phrases, and referring expressions) in the caption to their corresponding image regions.
More details can be found in the [paper](https://arxiv.org/abs/2306.14824).


### Download Data
- [GrIT-20M](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/data/grit_coyo.jsonl?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D): The split contains about 20M grounded image-caption pairs processed from COYO-700M. We also release it on [huggingface](https://huggingface.co/datasets/zzliang/GRIT).

The format of data instance is:

```python
{
  'clip_similarity_vitb32': 0.353271484375, 
  'clip_similarity_vitl14': 0.2958984375, 
  'id': 1795296605919, 
  'url': "https://www.thestrapsaver.com/wp-content/uploads/customerservice-1.jpg", 
  'caption': 'a wire hanger with a paper cover that reads we heart our customers', 
  'width': 1024, 
  'height': 693, 
  'noun_chunks': [[19, 32, 0.019644069503434333, 0.31054004033406574, 0.9622142865754519, 0.9603442351023356, 0.79298526], [0, 13, 0.019422357885505368, 0.027634161214033764, 0.9593302408854166, 0.969467560450236, 0.67520964]], 
  'ref_exps': [[19, 66, 0.019644069503434333, 0.31054004033406574, 0.9622142865754519, 0.9603442351023356, 0.79298526], [0, 66, 0.019422357885505368, 0.027634161214033764, 0.9593302408854166, 0.969467560450236, 0.67520964]]
}

```
- `clip_similarity_vitb32`: The cosine similarity between text and image(ViT-B/32) embeddings by [OpenAI CLIP](https://github.com/openai/CLIP), provided by COYO-700M.
- `clip_similarity_vitl14`: The cosine similarity between text and image(ViT-L/14) embeddings by [OpenAI CLIP](https://github.com/openai/CLIP), provided by COYO-700M.
- `id`: Unique 64-bit integer ID in COYO-700M.
- `url`: The image URL.
- `caption`: The corresponding caption.
- `width`: The width of the image.
- `height`: The height of the image.
- `noun_chunks`: The noun chunks (extracted by [spaCy](https://spacy.io/)) that have associated bounding boxes (predicted by [GLIP](https://github.com/microsoft/GLIP)). The items in the children list respectively represent 'Start of the noun chunk in caption', 'End of the noun chunk in caption', 'normalized x_min', 'normalized y_min', 'normalized x_max', 'normalized y_max', 'confidence score'.
- `ref_exps`: The corresponding referring expressions. If a noun chunk has no expansion, we just copy it. 

Run the following commands to visualize it:
```bash
wget -O /tmp/grit_coyo.jsonl "https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/data/grit_coyo.jsonl?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D"

python data/visualize_grit.py
```

We recommend using [img2dataset](https://github.com/rom1504/img2dataset) to download images, as detailed [here](https://huggingface.co/datasets/zzliang/GRIT#download-image).

## Evaluation

### 1. Phrase grounding 
We evaluate phrase grounding task on [Flickr30k Entities](https://github.com/BryanPlummer/flickr30k_entities) under zero-shot setting:
| Model | Recall@1 on val split | Recall@1 on test split | 
| ----- | --------------------- | ---------------------- |
| Kosmos-2 | 77.8 | 78.7 |

More results and evaluation code can be found in [evaluation/flickr/README.md](evaluation/flickr_entities/README.md)

### 2. Referring expression comprehension
We evaluate referring expression comprehension task on RefCOCO, RefCOCO+, and RefCOCOg under zero-shot setting. We report accuracy metric here.

| Model | RefCOCO val | RefCOCO testA| RefCOCO testB | RefCOCO+ val | RefCOCO+ testA| RefCOCO+ testB | RefCOCOg val | RefCOCOg test|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kosmos-2 | 52.32 | 57.42 | 47.26 | 45.48 | 50.73 | 42.24 | 60.57 | 61.65 |

More results and evaluation code can be found in [evaluation/refcoco/README.md](evaluation/refcoco/README.md)

### 3. Referring expression generation
We evaluate the referring expression generation task on RefCOCOg under zero-shot and few-shot settings. We report Meteor and CIDEr metrics here.

| Model | Setting | Meteor | CIDEr |
| --- | --- | --- | --- |
| Kosmos-2 | zero-shot | 12.2 | 60.3 |
| Kosmos-2 | few-shot (k=2) | 13.8 | 62.2 |
| Kosmos-2 | few-shot (k=4) | 14.1 | 62.2 |

We will release the evaluation code in [here](evaluation/).

### 4. Image captioning
We evaluate the image captioning task on Flickr30K Karpathy split test set under the zero-shot setting. We report the CIDEr metric here.

| Model | CIDEr on Flickr30K | 
| --- | --- |
| Flamingo-3B | 60.6 |
| Flamingo-9B | 61.5 | 
| Kosmos-1 | 67.1 |
| Kosmos-2 | 80.5 | 

We will release the evaluation code in [here](evaluation/).

### 5. Visual question answering
We evaluate the visual question-answering task on the test-dev set of VQAv2 under the zero-shot setting. We report VQA scores obtained from the VQAv2 evaluation server.

| Model | Accuracy on VQAv2 | 
| --- | --- |
| Flamingo-3B | 49.2 |
| Flamingo-9B | 51.8 | 
| Kosmos-1 | 51.0 |
| Kosmos-2 | 51.1 | 

We will release the evaluation code in [here](evaluation/). 

## Training

### Preparing dataset

#### GrIT
After downloading the data from [huggingface](https://huggingface.co/datasets/zzliang/GRIT) using img2dataset, you will obtain some tar files. After decompressing them, you can get the images and corresponding JSON files. Then, modify the file path in [prepare_grit.py](data/prepare_grit.py) and run this file to get the corresponding tsv files. If a tsv file is too large, you can split it into multiple ones yourself.

After processing all the tar files into tsv files, run [generate_config.py](data/generate_config.py) to get a config file, which stores the paths of the tsv files. In [train.sh](train.sh), change the `--laion-data-dir` to the config directory path.

#### Interleaved data
Interleaved image-text data also needs to be processed in this way. To be updated.

#### Text data
To be updated.

### Train script
After preparing the data, run the following command to train the model.
```bash
bash train.sh
```
More training/instruction-tuning tasks will be updated.

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

This repository is built using [torchscale](https://github.com/microsoft/torchscale), [fairseq](https://github.com/facebookresearch/fairseq), [openclip](https://github.com/mlfoundations/open_clip). We also would like to acknowledge the examples provided by [WHOOPS!](https://whoops-benchmark.github.io). We acknowledge [ydshieh](https://huggingface.co/ydshieh) at HuggingFace for [the online demo](https://huggingface.co/spaces/ydshieh/Kosmos-2) and [the HuggingFace's transformers implementation](https://huggingface.co/ydshieh/kosmos-2-patch14-224).


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using models, please submit a GitHub issue.
