# LayoutXLM (Document Foundation Model)
**Multimodal (text + layout/format + image) pre-training for multilingual [Document AI](https://www.microsoft.com/en-us/research/project/document-ai/)**

## Introduction

LayoutXLM is a multimodal pre-trained model for multilingual document understanding, which aims to bridge the language barriers for visually-rich document understanding. Experiment results show that it has significantly outperformed the existing SOTA cross-lingual pre-trained models on the XFUND dataset.

[LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836)
Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei, arXiv Preprint 2021

## Models
`layoutxlm-base` | [huggingface](https://huggingface.co/microsoft/layoutxlm-base)

## Fine-tuning Example on [XFUND](https://github.com/doc-analysis/XFUND)

### Installation

Please refer to [layoutlmft](../layoutlmft/README.md)

### Fine-tuning for Semantic Entity Recognition

```
cd layoutlmft
python -m torch.distributed.launch --nproc_per_node=4 examples/run_xfun_ser.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --output_dir /tmp/test-ner \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16
```

### Fine-tuning for Relation Extraction

```
cd layoutlmft
python -m torch.distributed.launch --nproc_per_node=4 examples/run_xfun_re.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --output_dir /tmp/test-ner \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 2500 \
        --per_device_train_batch_size 2 \
        --warmup_ratio 0.1 \
        --fp16
```

## Results on [XFUND](https://github.com/doc-analysis/XFUND)

###  Language-specific Finetuning

|                             | Model              | FUNSD      | ZH         | JA         | ES         | FR         | IT         | DE         | PT         | Avg.       |
| --------------------------- | ------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Semantic Entity Recognition | `xlm-roberta-base` | 0.667      | 0.8774     | 0.7761     | 0.6105     | 0.6743     | 0.6687     | 0.6814     | 0.6818     | 0.7047     |
|                             | `infoxlm-base`     | 0.6852     | 0.8868     | 0.7865     | 0.6230     | 0.7015     | 0.6751     | 0.7063     | 0.7008     | 0.7207     |
|                             | `layoutxlm-base`   | **0.794**  | **0.8924** | **0.7921** | **0.7550** | **0.7902** | **0.8082** | **0.8222** | **0.7903** | **0.8056** |
| Relation Extraction         | `xlm-roberta-base` | 0.2659     | 0.5105     | 0.5800     | 0.5295     | 0.4965     | 0.5305     | 0.5041     | 0.3982     | 0.4769     |
|                             | `infoxlm-base`     | 0.2920     | 0.5214     | 0.6000     | 0.5516     | 0.4913     | 0.5281     | 0.5262     | 0.4170     | 0.4910     |
|                             | `layoutxlm-base`   | **0.5483** | **0.7073** | **0.6963** | **0.6896** | **0.6353** | **0.6415** | **0.6551** | **0.5718** | **0.6432** |
### Zero-shot Transfer Learning

|     | Model              | FUNSD      | ZH         | JA         | ES         | FR         | IT         | DE         | PT         | Avg.       |
| --- | ------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| SER | `xlm-roberta-base` | 0.667      | 0.4144     | 0.3023     | 0.3055     | 0.371      | 0.2767     | 0.3286     | 0.3936     | 0.3824     |
|     | `infoxlm-base`     | 0.6852     | 0.4408     | 0.3603     | 0.3102     | 0.4021     | 0.2880     | 0.3587     | 0.4502     | 0.4119     |
|     | `layoutxlm-base`   | **0.794**  | **0.6019** | **0.4715** | **0.4565** | **0.5757** | **0.4846** | **0.5252** | **0.539**  | **0.5561** |
| RE  | `xlm-roberta-base` | 0.2659     | 0.1601     | 0.2611     | 0.2440     | 0.2240     | 0.2374     | 0.2288     | 0.1996     | 0.2276     |
|     | `infoxlm-base`     | 0.2920     | 0.2405     | 0.2851     | 0.2481     | 0.2454     | 0.2193     | 0.2027     | 0.2049     | 0.2423     |
|     | `layoutxlm-base`   | **0.5483** | **0.4494** | **0.4408** | **0.4708** | **0.4416** | **0.4090** | **0.3820** | **0.3685** | **0.4388** |

### Multitask Fine-tuning



|     | Model              | FUNSD      | ZH         | JA         | ES         | FR         | IT         | DE         | PT         | Avg.       |
| --- | ------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| SER | `xlm-roberta-base` | 0.6633     | 0.883      | 0.7786     | 0.6223     | 0.7035     | 0.6814     | 0.7146     | 0.6726     | 0.7149     |
|     | `infoxlm-base`     | 0.6538     | 0.8741     | 0.7855     | 0.5979     | 0.7057     | 0.6826     | 0.7055     | 0.6796     | 0.7106     |
|     | `layoutxlm-base`   | **0.7924** | **0.8973** | **0.7964** | **0.7798** | **0.8173** | **0.821**  | **0.8322** | **0.8241** | **0.8201** |
| RE  | `xlm-roberta-base` | 0.3638     | 0.6797     | 0.6829     | 0.6828     | 0.6727     | 0.6937     | 0.6887     | 0.6082     | 0.6341     |
|     | `infoxlm-base`     | 0.3699     | 0.6493     | 0.6473     | 0.6828     | 0.6831     | 0.6690     | 0.6384     | 0.5763     | 0.6145     |
|     | `layoutxlm-base`   | **0.6671** | **0.8241** | **0.8142** | **0.8104** | **0.8221** | **0.8310** | **0.7854** | **0.7044** | **0.7823** |

## Citation

If you find LayoutXLM useful in your research, please cite the following paper:

``` latex
@article{Xu2020LayoutXLMMP,
  title         = {LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding},
  author        = {Yiheng Xu and Tengchao Lv and Lei Cui and Guoxin Wang and Yijuan Lu and Dinei Florencio and Cha Zhang and Furu Wei},
  year          = {2021},
  eprint        = {2104.08836},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

## License

The content of this project itself is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### Contact Information

For help or issues using LayoutXLM, please submit a GitHub issue.

For other communications related to LayoutXLM, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).

