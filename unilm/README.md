# UniLM
**Unified pre-training for language understanding (NLU) and generation (NLG)**

**UniLM v2** ```New``` (February, 2020): "[UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training](https://arxiv.org/abs/2002.12804)".

**[UniLM v1](https://github.com/microsoft/unilm/tree/master/unilm-v1)** (September 30th, 2019): the code and pre-trained models for the NeurIPS 2019 paper entitled "[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)".

# Pre-trained Models
We release both base-size and large-size **uncased** **[UniLMv2](https://arxiv.org/abs/2002.12804)** models pre-trained with the same 160G text corpora as in [RoBERTa](https://arxiv.org/abs/1907.11692). The models are trained using the same WordPiece vocabulary as [BERT](https://github.com/google-research/bert). 
The links to the pre-trained models:
- [unilm2-large-cased]: 24-layer, 1024-hidden, 16-heads, 340M parameters
- [unilm2-base-cased]: 12-layer, 768-hidden, 12-heads, 110M parameters

# Fine-tuning on NLG Tasks
## Abstractive Summarization
We report the finetuning results of UniLMv2 for abstractive summarization on the following widely-used benchmarks datasets.
- [CNN / Daily Mail](https://github.com/harvardnlp/sent-summary): [the non-anonymized version of CNN/DailyMail (See et al., 2017)](https://arxiv.org/abs/1704.04368) which contains news articles (39 sentences on average) paired with multi-sentence summaries.
- [Gigaword](https://github.com/harvardnlp/sent-summary): a headline generation dataset which contains 3.8M examples.
- [XSum](https://github.com/EdinburghNLP/XSum): consists of 226,711 Wayback archived BBC articles ranging over almost a decade (2010 to 2017) and covering a wide variety of domains (e.g., News, Politics, Sports, Weather, Business, Technology, Science, Health, Family, Education, Entertainment and Arts).

#### BASE-size pre-trained models

|     Model                                                          |     # Param   |     Corpus   |    CNN/DailyMail                  |                      XSum                    |                    Gigaword                  |
|--------------------------------------------------------------------|---------------|--------------|:--------------------------:|:--------------------------------------------:|:--------------------------------------------:|
|                                                                    |               |              |                 RG-1/RG-2/RG-L               |                 RG-1/RG-2/RG-L               |                 RG-1/RG-2/RG-L               |
| [PtrNet (See et al., 2017)](https://arxiv.org/pdf/1704.04368.pdf)                      | -             | -            |               39.53/17.28/36.38              |               28.10/8.02/21.72               |                       -                      |
| [MASS (Song et al., 2019)](http://proceedings.mlr.press/v97/song19d/song19d.pdf) | 123M          | -            |               42.12/19.50/39.01              |               39.75/17.24/31.95              |               38.73/19.71/35.96              |
| [BertSumAbs (Liu & Lapata, 2019)](https://arxiv.org/pdf/1908.08345.pdf)           | 156M          | 16GB         |               41.72/19.39/38.76              |               38.76/16.33/31.15              |                       -                      |
| [ERNIE-GEN (Xiao et al., 2020)](https://arxiv.org/pdf/2001.11314.pdf) | 110M          | 16GB         |               42.30/19.92/39.68              |                       -                      |               38.83/20.04/36.20              |
| [T5-base (Raffel et al., 2019)](https://arxiv.org/pdf/1910.10683.pdf)         | 220M          | 750GB        |               42.05/20.34/39.40              |                       -                      |                       -                      |
| **UniLMv2 Base**                                       | 110M          | 160GB        | **43.87/20.99/40.95** |             **44.51/21.53/36.62**            |                       -                      |

#### LARGE-size pre-trained models

|     Model                                               |     # Param   |     Corpus   |      CNN/DailyMail       |             XSum         |                    Gigaword                  |
|---------------------------------------------------------|---------------|--------------|:------------------------:|:------------------------:|:------------------------:|
|                                                         |               |              |     RG-1/RG-2/RG-L               |                 RG-1/RG-2/RG-L               |                 RG-1/RG-2/RG-L               |
| [UniLM (Dong et al., 2019)](https://arxiv.org/pdf/1905.03197.pdf)  | 340M          | 16GB         |     43.08/20.43/40.34              |                       -                      |               38.90/20.05/36.00              |
| [ERNIE-GEN (Xiao et al., 2020)](https://arxiv.org/pdf/2001.11314.pdf) | 340M          | 16GB         |     44.02/21.17/41.26              |                       -                      |               39.25/20.25/36.53              |
| [BART (Raffel et al., 2019)](https://arxiv.org/pdf/1910.13461.pdf)                                 | 400M          | 160GB        |     44.16/21.28/40.90              |               45.14/22.27/37.25              |                       -                      |
| [ProphetNet (Yan et al., 2020)](https://arxiv.org/pdf/2001.04063.pdf)      | 400M          | 160GB        |     44.20/21.17/41.30              |                       -                      |               39.51/20.42/36.69              |
| [PEGASUSLARGE (C4) (Zhang et al., 2019)](https://arxiv.org/pdf/1912.08777.pdf)                | 568M          | 750GB        |     43.90/21.20/40.76              |               45.20/22.06/36.99              |               38.75/19.96/36.14              |
| [PEGASUSLARGE (HugeNews) (Zhang et al., 2019)](https://arxiv.org/pdf/1912.08777.pdf)           | 568M          | 3800GB       |     44.17/21.47/41.11              |             47.21/**24.56**/39.25            |               39.12/19.86/36.24              |
| [T5-11B (Raffel et al., 2019)](https://arxiv.org/pdf/1910.10683.pdf)                       | 11B           | 750GB        |     43.52/21.55/40.69              |                       -                      |                       -                      |
| **UniLMv2 Large**                                         | 340M          | 160GB        |   **44.79/21.98/41.93**  | **47.58**/24.35/**39.50** |             **39.73/20.70/36.89**            |

### Question Generation - [SQuAD](https://arxiv.org/abs/1806.03822)

We present the results following the same [data split](https://github.com/xinyadu/nqg/tree/master/data) and [evaluation scripts](https://github.com/xinyadu/nqg/tree/master/qgevalcap) as in [(Du et al., 2017)](https://arxiv.org/pdf/1705.00106.pdf). We also report the results following the data split as in [(Zhao et al., 2018)](https://aclweb.org/anthology/D18-1424), which uses the reversed dev-test setup.

| Model                                               | # Param | Corpus |   Official Split    |   Reversed Split    |
|-----------------------------------------------------|---------|--------|:-------------------:|:-------------------:|
|                                                     |         |        |   BLEU-4/MTR/RG-L   |   BLEU-4/MTR/RG-L   |
| [(Du & Cardie, 2018)](https://www.aclweb.org/anthology/P18-1177.pdf) | -       | -      |    15.16/19.12/-    |          -          |
| [(Zhao et al., 2018)](https://www.aclweb.org/anthology/D18-1424.pdf) | -       | -      |          -          |  16.38/20.25/44.48  |
| [(Zhang & Bansal, 2019)](https://arxiv.org/pdf/1909.06356.pdf) | -       | -      |  18.37/22.65/46.68  |  20.76/24.20/48.91  |
| [ERNIE-GEN base (Xiao et al., 2020)](https://arxiv.org/pdf/2001.11314.pdf)  | 110M    | 16GB   |  22.28/25.13/50.58  |  23.52/25.61/51.45  |
| [UniLM (Dong et al., 2019)](https://arxiv.org/pdf/1905.03197.pdf)        | 340M    | 16GB   |  22.12/25.06/51.07  |  23.75/25.61/52.04  |
| [ERNIE-GEN large (Xiao et al., 2020)](https://arxiv.org/pdf/2001.11314.pdf) | 340M    | 16GB   |  24.03/26.31/52.36  |  25.57/26.89/53.31  |
| [ProphetNet (Yan et al., 2020)](https://arxiv.org/pdf/2001.04063.pdf) | 400M    | 16GB   |  25.01/26.83/52.57  |  26.72/27.64/53.79  |
| **UniLMv2 Base**                                   | 110M    | 160GB  |24.70/26.33/52.13|26.30/27.09/53.19|
| **UniLMv2 Large**                                  | 340M    | 160GB  |**26.15/27.31/53.40**|**27.28/28.04/54.33**|

# Fine-tuning on NLU Tasks

## Citation

If you find UniLM useful in your work, you can cite the following paper:
```
@inproceedings{unilmv2,
    title={UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training},
    author={Bao, Hangbo and Dong, Li and Wei, Furu and Wang, Wenhui and Yang, Nan and Liu, Xiaodong and Wang, Yu and Piao, Songhao and Gao, Jianfeng and Zhou, Ming and Hon, Hsiao-Wuen},
    year={2020},
    booktitle = "Preprint"
}
```

## Acknowledgments
Our code is based on [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0). We thank the authors for their wonderful open-source efforts.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using UniLM, please submit a GitHub issue.

For other communications related to UniLM, please contact Li Dong (`lidong1@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).
