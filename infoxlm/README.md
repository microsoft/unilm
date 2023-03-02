# InfoXLM
**Cross-Lingual Language Model Pre-training**

## Overview

Code for pretraining cross-lingual language models. This repo provides implementations of various cross-lingual language models, including:

- **InfoXLM** (NAACL 2021, [paper](https://arxiv.org/pdf/2007.07834.pdf), [repo](https://github.com/microsoft/unilm/tree/master/infoxlm), [model](https://huggingface.co/microsoft/infoxlm-base)) InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training.

- **XLM-E** (arXiv 2021, [paper](https://arxiv.org/pdf/2106.16138.pdf)) XLM-E: Cross-lingual Language Model Pre-training via ELECTRA

- **XLM-Align** (ACL 2021, [paper](https://aclanthology.org/2021.acl-long.265/), [repo](https://github.com/CZWin32768/XLM-Align), [model](https://huggingface.co/microsoft/xlm-align-base)) Improving Pretrained Cross-Lingual Language Models via Self-Labeled Word Alignment

- **mBERT** Pretraining BERT on multilingual text with the masked language modeling (MLM) task.

- **XLM** Pretraining Transformer encoder with masked language modeling (MLM) and translation language modeling (TLM).

The following models will be also added to this repo ASAP:

- **XNLG** (AAAI 2020, [paper](https://arxiv.org/pdf/1909.10481.pdf), [repo](https://github.com/CZWin32768/XNLG)) multilingual/cross-lingual pre-trained model for natural language generation, e.g., finetuning XNLG with English abstractive summarization (AS) data and directly performing French AS or even Chinese-French AS.

- **mT6** ([paper](https://arxiv.org/abs/2104.08692)) mT6: Multilingual Pretrained Text-to-Text Transformer with Translation Pairs

## How to Use

### From Hugging Face model hub

We provide the models in Hugging Face format, so you can use the model directly with Hugging Face API:

**XLM-Align**
```python
model = AutoModel.from_pretrained("microsoft/xlm-align-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
```

**InfoXLM-base**
```python
model = AutoModel.from_pretrained("microsoft/infoxlm-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
```

**InfoXLM-large**
```python
model = AutoModel.from_pretrained("microsoft/infoxlm-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
```

### Finetuning on end tasks

Our models use the same vocabulary, tokenizer, and architecture with XLM-Roberta. So you can directly use the existing codes for finetuning XLM-R, **just by replacing the model name from `xlm-roberta-base` to `microsoft/xlm-align-base`, `microsoft/infoxlm-base`, or `microsoft/infoxlm-base`**.

For example, you can evaluate our model with [xTune](https://github.com/bozheng-hit/xTune)[3] on the XTREME benchmark.

## Pretraining

### Environment

The recommended way to run the code is using docker:
```bash
docker run -it --rm --runtime=nvidia --ipc=host --privileged pytorch/pytorch:1.4-cuda10.1-cudnn7-devel bash
```

The docker is initialized by:
```bash
. .bashrc
apt-get update
apt-get install -y vim wget ssh

PWD_DIR=$(pwd)
cd $(mktemp -d)
# install apex
git clone -q https://github.com/NVIDIA/apex.git
cd apex
git reset --hard 11faaca7c8ff7a7ba6d55854a9ee2689784f7ca5
python setup.py install --user --cuda_ext --cpp_ext
cd ..
cd $PWD_DIR

git clone https://github.com/microsoft/unilm
cd unilm/infoxlm

# install fairseq https://github.com/CZWin32768/fairseq/tree/czw
pip install --user --editable ./fairseq

# install infoxlm
pip install --user --editable ./src-infoxlm
```

### Prepare Training Data

All the training data are preprocessed into fairseq mmap format.

**Prepare MLM data**

The MLM training data should be preprocessed into token blocks with the length of 512.

**Step1**: Prepare training data in text format with one sentence per line. The text file should contain multilingual unlabeled text.

Example:

```
This is just an example.
Bonjour!
今天天气怎么样？
...
```

**Step2**: Convert to token blocks with the length of 512 in fairseq `mmap` format

Example:
```
<s> This is just an example . </s> Bonjour ! </s> 今天 天气 怎么样 ？ </s>
...
```


Command:
```
python ./tools/txt2bin.py \
--model_name microsoft/xlm-align-base \
--input /path/to/text.txt \
--output /path/to/output/dir
```

**Step3**: Put the `dict.txt` to the data dir. (Note: In InfoXLM and XLM-Align, we use the same `dict.txt` as [the dict file of XLM-R](https://github.com/pytorch/fairseq/tree/master/examples/xlmr). )


**Prepare TLM Data**

**Step1**: Prepare parallel data in text format with one sentence per line.

Example:

At en-zh.en.txt
```
This is just an example.
Hello world!
...
```

At en-zh.zh.txt
```
这只是一个例子。
你好世界！
...
```

**Step2**: Concatenate the parallel sentences into fairseq `mmap` format. 

Example:

```
<s> This is just an example . <\s> 这 只是 一个 例 子 。 <\s>
<s> Hello world ! <\s> 你好 世界 ！<\s>
...
```

Command:
```
python ./tools/para2bin.py \
--model_name microsoft/xlm-align-base \
--input_src /path/to/src-trg.src.txt \
--input_trg /path/to/src-trg.trg.txt \
--output /path/to/output/dir
```

**Prepare XlCo Data**


**Step1**: Prepare parallel data in text format with one sentence per line.

**Step2**: Alternately store the token indices of the two input files, and save the resulting dataset into fairseq `mmap` format.

Example:
```
<s> This is just an example . </s>
<s> 这 只是 一个 例 子 。 </s>
<s> Hello world ! </s>
<s> 你好 世界 ！ </s>
...
```

Command:
```
python ./tools/para2bin.py \
--model_name microsoft/xlm-align-base \
--input_src /path/to/src-trg.src.txt \
--input_trg /path/to/src-trg.trg.txt \
--output /path/to/output/dir
```

### Pretrain InfoXLM

Continue-train InfoXLM-base from XLM-R-base

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python src-infoxlm/train.py ${MLM_DATA_DIR} \
--task infoxlm --criterion xlco \
--tlm_data ${TLM_DATA_DIR} \
--xlco_data ${XLCO_DATA_DIR} \
--arch infoxlm_base --sample-break-mode complete --tokens-per-sample 512 \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
--lr-scheduler polynomial_decay --lr 0.0002 --warmup-updates 10000 \
--total-num-update 200000 --max-update 200000 \
--dropout 0.0 --attention-dropout 0.0 --weight-decay 0.01 \
--max-sentences 16 --update-freq 16 \
--log-format simple --log-interval 1 --disable-validation \
--save-interval-updates 5000 --no-epoch-checkpoints \
--fp16 --fp16-init-scale 128 --fp16-scale-window 128 --min-loss-scale 0.0001 \
--seed 1 \
--save-dir .${SAVE_DIR}/ \
--tensorboard-logdir .${SAVE_DIR}/tb-log \
--roberta-model-path /path/to/model.pt \
--num-workers 4 --ddp-backend=c10d --distributed-no-spawn \
--xlco_layer 8 --xlco_queue_size 131072 --xlco_lambda 1.0 \
--xlco_momentum constant,0.9999 --use_proj
```

- `${MLM_DATA_DIR}`: directory to mlm training data.
- `${SAVE_DIR}`: checkpoints are saved in this folder.
- `--max-sentences 8`: batch size per GPU.
- `--update-freq 32`: gradient accumulation steps. (total batch size = TOTAL_NUM_GPU x max-sentences x update-freq = 8 x 16 x 16 = 2048)
- `--roberta-model-path`: the checkpoint path to an existing roberta model (as the initialization of the current model). For learning from scratch, remove this line. The `model.pt` file of XLM-R can be downloaded from [here](https://github.com/pytorch/fairseq/tree/master/examples/xlmr)
- `--xlco_layer`: the layer to perform cross-lingual contrast (XlCo)
- `--xlco_lambda`: the weight of XlCo loss


### Pretrain XLM-Align

Continue-train XLM-Align-base from XLM-R-base

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python src-infoxlm/train.py ${MLM_DATA_DIR} \
--task xlm_align --criterion dwa_mlm_tlm \
--tlm_data ${TLM_DATA_DIR} \
--arch xlm_align_base --sample-break-mode complete --tokens-per-sample 512 \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 \
--clip-norm 1.0 --lr-scheduler polynomial_decay --lr 0.0002 \
--warmup-updates 10000 --total-num-update 200000 --max-update 200000 \
--dropout 0.0 --attention-dropout 0.0 --weight-decay 0.01 \
--max-sentences 16 --update-freq 16 --log-format simple \
--log-interval 1 --disable-validation --save-interval-updates 5000 --no-epoch-checkpoints \
--fp16 --fp16-init-scale 128 --fp16-scale-window 128 --min-loss-scale 0.0001 \
--seed 1 \
--save-dir .${SAVE_DIR} \
--tensorboard-logdir .${SAVE_DIR}/tb-log \
--roberta-model-path /path/to/model.pt \
--num-workers 2 --ddp-backend=c10d --distributed-no-spawn \
--wa_layer 10 --wa_max_count 2 --sinkhorn_iter 2
```

- `${MLM_DATA_DIR}`: directory to mlm training data.
- `${SAVE_DIR}`: checkpoints are saved in this folder.
- `--max-sentences 8`: batch size per GPU.
- `--update-freq 32`: gradient accumulation steps. (total batch size = TOTAL_NUM_GPU x max-sentences x update-freq = 8 x 16 x 16 = 2048)
- `--roberta-model-path`: the checkpoint path to an existing roberta model (as the initialization of the current model). For learning from scratch, remove this line.
- `--wa_layer`: the layer to perform word alignment self-labeling
- `--wa_max_count`: the number of iterative alignment filtering
- `--sinkhorn_iter`: the number of the iteration in Sinkhorn's algorithm


### Pretrain MLM

Continue-train MLM / mBert from XLM-R-base

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python src-infoxlm/train.py ${MLM_DATA_DIR} \
--task mlm --criterion masked_lm \
--arch reload_roberta_base --sample-break-mode complete --tokens-per-sample 512 \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
--lr-scheduler polynomial_decay --lr 0.0002 --warmup-updates 10000 \
--total-num-update 200000 --max-update 200000 \
--dropout 0.0 --attention-dropout 0.0 --weight-decay 0.01 \
--max-sentences 32 --update-freq 8 \
--log-format simple --log-interval 1 --disable-validation \
--save-interval-updates 5000 --no-epoch-checkpoints \
--fp16 --fp16-init-scale 128 --fp16-scale-window 128 --min-loss-scale 0.0001 \
--seed 1 \
--save-dir .${SAVE_DIR}/ \
--tensorboard-logdir .${SAVE_DIR}/tb-log \
--roberta-model-path /path/to/model.pt \
--num-workers 2 --ddp-backend=c10d --distributed-no-spawn
```

Pretraining MLM / mBERT from scratch

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python src-infoxlm/train.py ${MLM_DATA_DIR} \
--task mlm --criterion masked_lm \
--arch reload_roberta_base --sample-break-mode complete --tokens-per-sample 512 \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
--lr-scheduler polynomial_decay --lr 0.0001 --warmup-updates 10000 \
--total-num-update 1000000 --max-update 1000000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-sentences 32 --update-freq 1 \
--log-format simple --log-interval 1 --disable-validation \
--save-interval-updates 5000 --no-epoch-checkpoints \
--fp16 --fp16-init-scale 128 --fp16-scale-window 128 --min-loss-scale 0.0001 \
--seed 1 \
--save-dir .${SAVE_DIR}/ \
--tensorboard-logdir .${SAVE_DIR}/tb-log \
--num-workers 2 --ddp-backend=c10d --distributed-no-spawn
```

### Pretrain MLM+TLM

Continue-train MLM+TLM from XLM-R-base

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python src-infoxlm/train.py ${MLM_DATA_DIR} \
--tlm_data ${TLM_DATA_DIR} \
--task tlm --criterion masked_lm \
--arch reload_roberta_base --sample-break-mode complete --tokens-per-sample 512 \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
--lr-scheduler polynomial_decay --lr 0.0002 --warmup-updates 10000 \
--total-num-update 200000 --max-update 200000 \
--dropout 0.0 --attention-dropout 0.0 --weight-decay 0.01 \
--max-sentences 32 --update-freq 8 \
--log-format simple --log-interval 1 --disable-validation \
--save-interval-updates 5000 --no-epoch-checkpoints \
--fp16 --fp16-init-scale 128 --fp16-scale-window 128 --min-loss-scale 0.0001 \
--seed 1 \
--save-dir .${SAVE_DIR}/ \
--tensorboard-logdir .${SAVE_DIR}/tb-log \
--roberta-model-path /path/to/model.pt \
--num-workers 2 --ddp-backend=c10d --distributed-no-spawn
```

Pretraining MLM+TLM from scratch

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python src-infoxlm/train.py ${MLM_DATA_DIR} \
--tlm_data ${TLM_DATA_DIR} \
--task tlm --criterion masked_lm \
--arch reload_roberta_base --sample-break-mode complete --tokens-per-sample 512 \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
--lr-scheduler polynomial_decay --lr 0.0001 --warmup-updates 10000 \
--total-num-update 1000000 --max-update 1000000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-sentences 32 --update-freq 1 \
--log-format simple --log-interval 1 --disable-validation \
--save-interval-updates 5000 --no-epoch-checkpoints \
--fp16 --fp16-init-scale 128 --fp16-scale-window 128 --min-loss-scale 0.0001 \
--seed 1 \
--save-dir .${SAVE_DIR}/ \
--tensorboard-logdir .${SAVE_DIR}/tb-log \
--num-workers 2 --ddp-backend=c10d --distributed-no-spawn
```

## References

Please cite the papers if you found the resources in this repository useful.

[1] **XLM-Align** (ACL 2021, [paper](https://aclanthology.org/2021.acl-long.265/), [repo](https://github.com/CZWin32768/XLM-Align), [model](https://huggingface.co/microsoft/xlm-align-base)) Improving Pretrained Cross-Lingual Language Models via Self-Labeled Word Alignment

```
@inproceedings{xlmalign,
  title = "Improving Pretrained Cross-Lingual Language Models via Self-Labeled Word Alignment",
  author={Zewen Chi and Li Dong and Bo Zheng and Shaohan Huang and Xian-Ling Mao and Heyan Huang and Furu Wei},
  booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
  month = aug,
  year = "2021",
  address = "Online",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.acl-long.265",
  doi = "10.18653/v1/2021.acl-long.265",
  pages = "3418--3430",}
```

[2] **InfoXLM** (NAACL 2021, [paper](https://arxiv.org/pdf/2007.07834.pdf), [repo](https://github.com/microsoft/unilm/tree/master/infoxlm), [model](https://huggingface.co/microsoft/infoxlm-base)) InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training.

```
@inproceedings{chi-etal-2021-infoxlm,
  title = "{I}nfo{XLM}: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training",
  author={Chi, Zewen and Dong, Li and Wei, Furu and Yang, Nan and Singhal, Saksham and Wang, Wenhui and Song, Xia and Mao, Xian-Ling and Huang, Heyan and Zhou, Ming},
  booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
  month = jun,
  year = "2021",
  address = "Online",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.naacl-main.280",
  doi = "10.18653/v1/2021.naacl-main.280",
  pages = "3576--3588",}
```

[3] **xTune** (ACL 2021, [paper](https://arxiv.org/pdf/2106.08226.pdf), [repo](https://github.com/bozheng-hit/xTune)) Consistency Regularization for Cross-Lingual Fine-Tuning.

```
@inproceedings{zheng-etal-2021-consistency,
    title = "Consistency Regularization for Cross-Lingual Fine-Tuning",
    author = {Bo Zheng, Li Dong, Shaohan Huang, Wenhui Wang, Zewen Chi, Saksham Singhal, Wanxiang Che, Ting Liu, Xia Song, Furu Wei},
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.264",
    doi = "10.18653/v1/2021.acl-long.264",
    pages = "3403--3417",
}
```

[4] **XLM-E** (arXiv 2021, [paper](https://arxiv.org/pdf/2106.16138.pdf)) XLM-E: Cross-lingual Language Model Pre-training via ELECTRA

```
@misc{chi2021xlme,
      title={XLM-E: Cross-lingual Language Model Pre-training via ELECTRA}, 
      author={Zewen Chi and Shaohan Huang and Li Dong and Shuming Ma and Saksham Singhal and Payal Bajaj and Xia Song and Furu Wei},
      year={2021},
      eprint={2106.16138},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using InfoXLM, please submit a GitHub issue.

For other communications related to InfoXLM, please contact Li Dong (`lidong1@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).

