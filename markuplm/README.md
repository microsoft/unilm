# MarkupLM

**Multimodal (text +markup language) pre-training for [Document AI](https://www.microsoft.com/en-us/research/project/document-ai/)**

## Introduction

MarkupLM is a simple but effective multi-modal pre-training method of text and markup language for visually-rich document understanding and information extraction tasks, such as webpage QA and webpage information extraction. MarkupLM achieves the SOTA results on multiple datasets. For more details, please refer to our paper:

[MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding](https://arxiv.org/abs/2110.08518)  Junlong Li, Yiheng Xu, Lei Cui, Furu Wei, [Preprint](https://github.com/microsoft/unilm/tree/master/markuplm#)

The overview of our framework is as follows:
<div align="center">
<img src="https://user-images.githubusercontent.com/45759388/142979309-9b3ba8ce-d76c-482a-8ded-f837037e9e81.PNG" width="100%" height="100%" />
</div>

And the core XPath Embedding Layer is as follows:
<div align="center">
<img src="https://user-images.githubusercontent.com/45759388/142979238-22bc4910-1236-4c72-9292-47d613c39daa.PNG" width="70%" height="70%" />
</div>

## Release Notes

******* New Nov 22th, 2021: Initial release of pre-trained models and fine-tuning code for MarkupLM *******

## Pre-trained Models

We pre-train MarkupLM on a subset of the CommonCrawl dataset.

| Name  | HuggingFace |
| - | - | 
| MarkupLM-Base | [microsoft/markuplm-base](https://huggingface.co/microsoft/markuplm-base) |
| MarkupLM-Large | [microsoft/markuplm-large](https://huggingface.co/microsoft/markuplm-large) |

An example might be ``model = markuplm.from_pretrained("microsoft/markuplm-base")``.

## Installation

### Command

```
conda create -n markuplmft python=3.7
conda activate markuplmft
git clone https://github.com/microsoft/unilm.git
cd unilm
cd markuplm
pip install -r requirements.txt
pip install -e .
```

## Finetuning

### WebSRC

#### Prepare data

Download the dataset from the [official website](https://x-lance.github.io/WebSRC/).

Extract **release.zip** to **/Path/To/WebSRC**.

Download **dataset_split.json** from [this link](https://github.com/X-LANCE/WebSRC-Baseline/blob/master/data/dataset_split.json) and put it into **/Path/To/WebSRC**.

#### Generate dataset

```
cd ./examples/fine_tuning/run_websrc
python dataset_generation.py --root_dir /Path/To/WebSRC --version websrc1.0
```

#### Run

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py \
	--train_file /Path/To/WebSRC/websrc1.0_train_.json \
	--predict_file /Path/To/WebSRC/websrc1.0_dev_.json \
	--root_dir /Path/To/WebSRC \
	--model_name_or_path microsoft/markuplm-large \
	--output_dir /Your/Output/Path \
	--do_train \
	--do_eval \
	--eval_all_checkpoints \
	--per_gpu_train_batch_size 8 \
	--warmup_ratio 0.1 \
	--num_train_epochs 5
```

### SWDE

#### Prepare data

Download the dataset from the [official website](https://archive.codeplex.com/?p=swde).

Update: the above website is down, please use this [backup](http://web.archive.org/web/20210630013015/https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip).

Unzip **swde.zip**, and extract everything in **/sourceCode**, make sure we have folders like **auto / book / camera** ... under this directory, and we name this path as **/Path/To/SWDE**.

#### Generate dataset

```
cd ./examples/fine_tuning/run_swde

python pack_data.py \
	--input_swde_path /Path/To/SWDE \
	--output_pack_path /Path/To/SWDE/swde.pickle

python prepare_data.py \
	--input_groundtruth_path /Path/To/SWDE/groundtruth \
	--input_pickle_path /Path/To/SWDE/swde.pickle \
	--output_data_path /Path/To/Processed_SWDE
```

And the needed data is in **/Path/To/Processed_SWDE**.

#### Run

Take **seed=1, vertical=nbaplayer** as example.

```
CUDA_VISIBLE_DEVICES=0,1 python run.py \
	--root_dir /Path/To/Processed_SWDE \
	--vertical nbaplayer \
	--n_seed 1 \
	--n_pages 2000 \
	--prev_nodes_into_account 4 \
	--model_name_or_path microsoft/markuplm-base \
	--output_dir /Your/Output/Path \
	--do_train \
	--do_eval \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--num_train_epochs 10 \
	--learning_rate 2e-5 \
	--save_steps 1000000 \
	--warmup_ratio 0.1 \
	--overwrite_output_dir \
```

### Results

#### WebSRC (dev set)

Some of the baseline results are from [Chen et al., 2021](https://aclanthology.org/2021.emnlp-main.343/).

| Model | EM | F1 | POS |
| - | - | -| -|
|H-PLM (RoBERTa-Large) | 69.57| 74.13 | 85.93|
|H-PLM (ELECTRA-Large) |70.12 |74.14 | 86.33|
|V-PLM (ELECTRA-Large) |73.22 | 76.16|87.06 |
|**MarkupLM-Large** |**74.43** |**80.54** | **90.15**|

#### SWDE

The metric is **page-level F1**.

|Model \\ #Seed Sites| 1 | 2 | 3 | 4 | 5 |
|-|-|-|-|-|-|
|[Render-Full (Hao et al., 2011)](https://dl.acm.org/doi/10.1145/2009916.2010020)|84.30|86.00|86.80|88.40|88.60|
|[FreeDOM-Full (Lin et al., 2020)](https://dl.acm.org/doi/10.1145/3394486.3403153)|82.32|86.36|90.49|91.29|92.56|
|[SimpDOM (Zhou et al., 2021)](https://arxiv.org/abs/2101.02415)|83.06|88.96|91.63|92.84|93.75|
|**MarkupLM-Large**|**85.71**|**93.57**|**96.12**|**96.71**|**97.37**|

## Citation

If you find markupLM useful in your research, please cite the following paper:

<pre>
@article{li2021markuplm,
      title={MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding}, 
      author={Junlong Li and Yiheng Xu and Lei Cui and Furu Wei},
      year={2021},
      eprint={2110.08518},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
</pre>

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree. Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project. [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using MarkupLM, please submit a GitHub issue.

For other communications related to MarkupLM, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).

