# xTune

Code for ACL2021 paper [Consistency Regularization for Cross-Lingual Fine-Tuning](https://arxiv.org/pdf/2106.08226.pdf).
## Environment

DockerFile: `dancingsoul/pytorch:xTune`

Install the fine-tuning code: `pip install --user .`

## Data & Model Preparation

### XTREME Datasets  

1) Create a download folder with `mkdir -p download` in the root of this project. 
2) manually download `panx_dataset` (for NER) [here][2], (note that it will download as `AmazonPhotos.zip`) to the download directory.
3) run the following command to download the remaining datasets: `bash scripts/download_data.sh`
The code of downloading dataset from XTREME is from [xtreme offical repo][1].

Note that we keep the labels in test set for easier evaluation. To prevent accidental evaluation on the test sets while running experiments, the code of [xtreme offical repo][1] removes labels of the test data during pre-processing and changes the order of the test sentences for cross-lingual sentence retrieval. 
Replace `csv.writer(fout, delimiter='\t')` with `csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')` in utils_process.py if using XTREME official repo.

### Translations

XTREME provides translations for SQuAD v1.1 (only train and dev), MLQA, PAWS-X, TyDiQA-GoldP, XNLI, and XQuAD, which can be downloaded from [here][3]. The `xtreme_translations` folder should be moved to the download directory. 

The target language translations for panx and udpos are obtained with Google Translate, since they are not provided. Our processed version can be downloaded from [here][4]. It should be merged with the above `xtreme_translations` folder.

### Bi-lingual dictionaries

We obtain the bi-lingual dictionaries from the [MUSE][6] repo. For convenience, you can download them from [here][7] and move it to the download directory, i.e., `./download/dicts`.

### Models

XLM-Roberta is supported. We utilize the [huggingface][5] format, which can be downloaded with `bash scripts/download_model.sh`.

## Fine-tuning Usage

Our default settings were using Nvidia V100-32GB GPU cards. If there were out-of-memory errors, you can reduce `per_gpu_train_batch_size` while increasing `gradient_accumulation_steps`, or use multi-GPU training.

xTune consists of a two-stage training process.
- Stage 1: fine-tuning with example consistency on the English training set.
- Stage 2: fine-tuning with example consistency on the augmented training set and regularize model consistency with the model from Stage 1.

It's recommended to use both Stage 1 and Stage 2 for token-level tasks, such as sequential labeling, and question answering. For text classification, you can only use Stage 1 if the computation budget was limited.

```bash
bash ./scripts/train.sh [setting] [dataset] [model] [stage] [gpu] [data_dir] [output_dir]
```
where the options are described as follows:
- `[setting]`: `translate-train-all` (using input translation for the languages other than English) or `cross-lingual-transfer` (only using English for zero-shot cross-lingual transfer)
- `[dataset]`: dataset names in XTREME, i.e., `xnli`, `panx`, `pawsx`, `udpos`, `mlqa`, `tydiqa`, `xquad`
- `[model]`: `xlm-roberta-base`, `xlm-roberta-large`
- `[stage]`: `1` (first stage), `2` (second stage)
- `[gpu]`: used to set environment variable `CUDA_VISIBLE_DEVICES`
- `[data_dir]`: folder of training data
- `[output_dir]`: folder of fine-tuning output

## Examples: XTREME Tasks

### XNLI fine-tuning on English training set and translated training sets (`translate-train-all`)

```bash
# run stage 1 of xTune
bash ./scripts/train.sh translate-train-all xnli xlm-roberta-base 1
# run stage 2 of xTune (optional)
bash ./scripts/train.sh translate-train-all xnli xlm-roberta-base 2
```

### XNLI fine-tuning on English training set (`cross-lingual-transfer`)

```bash
# run stage 1 of xTune
bash ./scripts/train.sh cross-lingual-transfer xnli xlm-roberta-base 1
# run stage 2 of xTune (optional)
bash ./scripts/train.sh cross-lingual-transfer xnli xlm-roberta-base 2
```

## Paper
Please cite our paper `\cite{bo2021xtune}` if you found the resources in the repository useful.

```
@inproceedings{bo2021xtune,
author = {Bo Zheng, Li Dong, Shaohan Huang, Wenhui Wang, Zewen Chi, Saksham Singhal, Wanxiang Che, Ting Liu, Xia Song, Furu Wei},
booktitle = {Proceedings of ACL 2021},
title = {{Consistency Regularization for Cross-Lingual Fine-Tuning}},
year = {2021}
}
```

## Reference

1. https://github.com/google-research/xtreme
2. https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN?_encoding=UTF8&%2AVersion%2A=1&%2Aentries%2A=0&mgh=1
3. https://console.cloud.google.com/storage/browser/xtreme_translations
4. https://drive.google.com/drive/folders/1Rdbc0Us_4I5MpRCwLASxBwqSW8_dlF87?usp=sharing
5. https://github.com/huggingface/transformers/
6. https://github.com/facebookresearch/MUSE
7. https://drive.google.com/drive/folders/1k9rQinwUXicglA5oyzo9xtgqiuUVDkjT?usp=sharing
