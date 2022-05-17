# LayoutLM
**Multimodal (text + layout/format + image) pre-training for [Document AI](https://www.microsoft.com/en-us/research/project/document-ai/)**

- April, 2021: [LayoutXLM](https://github.com/microsoft/unilm/tree/master/layoutxlm) is coming by extending the LayoutLM into multilingual support! A multilingual form understanding benchmark [XFUND](https://github.com/doc-analysis/XFUND) is also introduced, which includes forms with human labeled key-value pairs in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese).
- December 29th, 2020: [LayoutLMv2](https://arxiv.org/abs/2012.14740) is coming with the new SOTA on a wide varierty of document AI tasks, including [DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1) and [SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) leaderboard.

## Introduction

LayoutLM is a simple but effective multi-modal pre-training method of text, layout and image for visually-rich document understanding and information extraction tasks, such as form understanding and receipt understanding. LayoutLM archives the SOTA results on multiple datasets. For more details, please refer to our paper: 

[LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)
Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, [KDD 2020](https://www.kdd.org/kdd2020/accepted-papers)

[LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740)
Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou, [ACL 2021](#)

[LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836)
Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei, [Preprint](#)

## Release Notes
**\*\*\*\*\* New Sep 27th, 2021: [**LayoutLM-cased**](https://huggingface.co/microsoft/layoutlm-base-cased) are on [HuggingFace](https://github.com/huggingface/transformers) \*\*\*\*\***

**\*\*\*\*\* New Aug 7th, 2020: Our new document understanding datasets, [TableBank](https://doc-analysis.github.io/tablebank-page/) (LREC 2020) and [DocBank](https://doc-analysis.github.io/docbank-page/) (COLING 2020), are now publicly available.\*\*\*\*\***

**\*\*\*\*\* New May 16th, 2020: Our LayoutLM paper has been accepted to KDD 2020 as a full paper in the research track\*\*\*\*\***

**\*\*\*\*\* New Feb 18th, 2020: Initial release of pre-trained models and fine-tuning code for LayoutLM v1 \*\*\*\*\***

## Pre-trained Model

We pre-train LayoutLM on IIT-CDIP Test Collection 1.0\* dataset. 

| name                    | #params | HuggingFace                                                  |
| ----------------------- | ------- | ------------------------------------------------------------ |
| LayoutLM-Base, Uncased  | 113M    | [Model Hub](https://huggingface.co/microsoft/layoutlm-base-uncased) |
| LayoutLM-Base, Cased    | 113M    | [Model Hub](https://huggingface.co/microsoft/layoutlm-base-cased) |
| LayoutLM-Large, Uncased | 343M    | [Model Hub](https://huggingface.co/microsoft/layoutlm-large-uncased) |



\*As some downstream datasets are the subsets of IIT-CDIP, we have carefully excluded the overlap portion from the pre-training data.

### Different Tokenizer
Note that LayoutLM-Base-Cased requires a different tokenizer, based on RobertaTokenizer. You can
initialize it as follows:

~~~
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlm-base-cased') 
~~~

## Fine-tuning Example on FUNSD

### Installation

Please refer to [layoutlmft](../layoutlmft/README.md)

### Command

```
cd layoutlmft
python -m torch.distributed.launch --nproc_per_node=4 examples/run_funsd.py \
        --model_name_or_path microsoft/layoutlm-base-uncased \
        --output_dir /tmp/test-ner \
        --do_train \
        --do_predict \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16
```



### Results

#### SROIE (field-level)


| Model                                                                                        | Hmean      |
| -------------------------------------------------------------------------------------------- | ---------- |
| BERT-Large                                                                                   | 90.99%     |
| RoBERTa-Large                                                                                | 92.80%     |
| [Ranking 1st in SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3)                  | 94.02%     |
| [**LayoutLM**](https://rrc.cvc.uab.es/?ch=13&com=evaluation&view=method_info&task=3&m=71448) | **96.04%** |

#### RVL-CDIP

| Model                                                                                | Accuracy   |
| ------------------------------------------------------------------------------------ | ---------- |
| BERT-Large                                                                           | 89.92%     |
| RoBERTa-Large                                                                        | 90.11%     |
| [VGG-16 (Afzal et al., 2017)](https://arxiv.org/abs/1704.03557)                      | 90.97%     |
| [Stacked CNN Ensemble (Das et al., 2018)](https://arxiv.org/abs/1801.09321)          | 92.21%     |
| [LadderNet (Sarkhel & Nandi, 2019)](https://www.ijcai.org/Proceedings/2019/0466.pdf) | 92.77%     |
| [Multimodal Ensemble (Dauphinee et al., 2019)](https://arxiv.org/abs/1912.04376)     | 93.07%     |
| **LayoutLM**                                                                         | **94.42%** |

#### FUNSD (field-level)

| Model         | Precision  | Recall     | F1         |
| ------------- | ---------- | ---------- | ---------- |
| BERT-Large    | 0.6113     | 0.7085     | 0.6563     |
| RoBERTa-Large | 0.6780     | 0.7391     | 0.7072     |
| **LayoutLM**  | **0.7677** | **0.8195** | **0.7927** |

## Citation

If you find LayoutLM useful in your research, please cite the following paper:

``` latex
@inproceedings{Xu2020LayoutLMPO,
  title   = {LayoutLM: Pre-training of Text and Layout for Document Image Understanding},
  author  = {Yiheng Xu and Minghao Li and Lei Cui and Shaohan Huang and Furu Wei and Ming Zhou},
  journal = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
  year    = {2020}
}
```

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using LayoutLM, please submit a GitHub issue.

For other communications related to LayoutLM, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).
