# UniLM
**Pre-trained models for natural language understanding (NLU) and generation (NLG) tasks**

The family of UniLM:
> [**UniLM**](https://github.com/microsoft/unilm/tree/master/unilm): **unified pre-training for language understanding and generation**

>  [**InfoXLM**](https://github.com/microsoft/unilm/tree/master/infoxlm) (```new```): **multilingual/cross-lingual pre-trained models for language understanding and generation**

>  [**MiniLM**](https://github.com/microsoft/unilm/tree/master/minilm): **small pre-trained models for language understanding and generation**

>[**LayoutLM**](https://github.com/microsoft/unilm/tree/master/layoutlm): **multimodal (text + layout/format + image) pre-training for document understanding** (e.g. scanned documents, PDF, etc.)

>  [**s2s-ft**](https://github.com/microsoft/unilm/tree/master/s2s-ft): **sequence-to-sequence fine-tuning toolkit**

## News
- July 16, 2020: [**InfoXLM** (Multilingual UniLM)](https://github.com/microsoft/unilm/tree/master/infoxlm) [arXiv](https://arxiv.org/pdf/2007.07834.pdf)
- June, 2020 (```NEW```): [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) was accepted by KDD 2020, and [UniLMv2](https://github.com/microsoft/unilm/tree/master/unilm) was accepted by ICML 2020.
- April 5, 2020: [**Multilingual MiniLM**](https://github.com/microsoft/unilm/tree/master/minilm) released!
- September, 2019: [UniLMv1](https://github.com/microsoft/unilm/tree/master/unilm-v1) was accepted by NeurIPS 2019.

## Release

**\*\*\*\*\* ```New February, 2020```: [UniLM v2](https://github.com/microsoft/unilm/tree/master/unilm) | [MiniLM v1](https://github.com/microsoft/unilm/tree/master/minilm) | [LayoutLM v1](https://github.com/microsoft/unilm/tree/master/layoutlm) | [s2s-ft v1](https://github.com/microsoft/unilm/tree/master/s2s-ft) release \*\*\*\*\***

- [x] [**LayoutLM 1.0**](https://github.com/microsoft/unilm/tree/master/layoutlm) (February 18, 2020): pre-trained models for document (image) understanding (e.g. receipts, forms, etc.) . It achieves new SOTA results in several downstream tasks, including form understanding (the FUNSD dataset from 70.72 to 79.27), receipt understanding (the [ICDAR 2019 SROIE leaderboard](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) from 94.02 to 95.24) and document image classification (the RVL-CDIP dataset from 93.07 to 94.42). "[LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) ```KDD 2020```"
- [x] [**s2s-ft 1.0**](https://github.com/microsoft/unilm/tree/master/s2s-ft) (February 26, 2020): A PyTorch package used to fine-tune pre-trained Transformers for sequence-to-sequence language generation. "[s2s-ft: Fine-Tuning Pre-Trained Transformers for Sequence-to-Sequence Learning](#)"
- [x] [**MiniLM 1.0**](https://github.com/microsoft/unilm/tree/master/minilm) (February 26, 2020): deep self-attention distillation is all you need (for task-agnostic knowledge distillation of pre-trained Transformers). MiniLM (12-layer, 384-hidden) achieves 2.7x speedup and comparable results over BERT-base (12-layer, 768-hidden) on NLU tasks as well as strong results on NLG tasks. The even smaller MiniLM (6-layer, 384-hidden) obtains 5.3x speedup and produces very competitive results. "[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)"
- [x] [**UniLM 2.0**](https://github.com/microsoft/unilm/tree/master/unilm) (February 28, 2020): **unified pre-training** of bi-directional LM (via autoencoding) and sequence-to-sequence LM (via partially autoregressive) w/ **Pseudo-Masked Language Model** for language understanding and generation. UniLM v2 achieves new SOTA in a wide range of natural language understanding and generation tasks. "[UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training](https://arxiv.org/abs/2002.12804) ```ICML 2020```"



**\*\*\*\*\* October 1st, 2019: UniLM v1 release \*\*\*\*\***

- [x] [**UniLM v1**](https://github.com/microsoft/unilm/tree/master/unilm-v1) (September 30, 2019): the code and pre-trained models for the ```NeurIPS 2019``` paper entitled "[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)". UniLM (v1) achieves the **new SOTA results** in **NLG** (especially **sequence-to-sequence generation**) tasks, including abstractive summarization (the Gigaword and CNN/DM datasets), question generation (the SQuAD QG dataset), etc. 

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using UniLM, please submit a GitHub issue.

For other communications related to UniLM, please contact Li Dong (`lidong1@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).

