# UniLM
**We develop pre-trained models for natural language understanding (NLU) and generation (NLG) tasks**

**\*\*\*\*\* ```New February, 2020```: [UniLM v2](https://github.com/microsoft/unilm/tree/master/unilm) | [MiniLM v1](https://github.com/microsoft/unilm/tree/master/minilm) | [LayoutLM v1](https://github.com/microsoft/unilm/tree/master/layoutlm) | [s2s-ft v1](https://github.com/microsoft/unilm/tree/master/s2s-ft) release \*\*\*\*\***

The family of UniLM:
> [**UniLM**](https://github.com/microsoft/unilm/tree/master/unilm): **unified pre-training for language understanding and generation**

>  [**MiniLM**](https://github.com/microsoft/unilm/tree/master/minilm) (```new```): **small pre-trained models for language understanding and generation**

>[**LayoutLM**](https://github.com/microsoft/unilm/tree/master/layoutlm) (```new```): **multimodal (text + layout/format + image) pre-training for document understanding** (e.g. scanned documents, PDF, etc.)

>  [**s2s-ft**](https://github.com/microsoft/unilm/tree/master/s2s-ft) (```new```): **sequence-to-sequence fine-tuning toolkit**

Update in this release:
- [x] [**LayoutLM 1.0**](https://github.com/microsoft/unilm/tree/master/layoutlm) (February 18, 2020): pre-trained models for document (image) understanding (e.g. receipts, forms, etc.) . It achieves new SOTA results in several downstream tasks, including form understanding (the FUNSD dataset from 70.72 to 79.27), receipt understanding (the [ICDAR 2019 SROIE leaderboard](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) from 94.02 to 95.24) and document image classification (the RVL-CDIP dataset from 93.07 to 94.42). "[LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)"
- [x] [**s2s-ft 1.0**](https://github.com/microsoft/unilm/tree/master/s2s-ft) (February 26, 2020): a pytorch package for finetuning pre-trained Transformers for language (sequence-to-sequence) generation. "[Fine-Tuning Bidirectional Transformers for Sequence-to-Sequence Learning](#)"
- [x] [**MiniLM 1.0**](https://github.com/microsoft/unilm/tree/master/minilm) (February 26, 2020): deep self-attention distillation is all you need (for task-agnostic knowledge distillation of pre-trained Transformers). We release MiniLM (12-layer, 384-hidden) w/ 2.7x speedup and comparable results over BERT-base (12-layer, 768-hidden) as well as strong results on NLG tasks. The even smaller MiniLM (6-layer, 384-hidden) w/ 5.3x speedup and can also produce very competitive results. "[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)"
- [ ] [**UniLM 2.0**](#)


**\*\*\*\*\* October 1st, 2019: UniLM v1 release \*\*\*\*\***

- [x] [**UniLM v1**](https://github.com/microsoft/unilm/tree/master/unilm-v1) (September 30, 2019): the code and pre-trained models for the NeurIPS 2019 paper entitled "[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)". UniLM (v1) achieves the **new SOTA results** in **NLG** (especially **sequence-to-sequence generation**) tasks/benchmarks, including abstractive summarization (the Gigaword and CNN/DM dataset), question generation (the SQuAD QG dataset), etc. 

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using UniLM, please submit a GitHub issue.

For other communications related to UniLM, please contact Li Dong (`lidong1@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).

