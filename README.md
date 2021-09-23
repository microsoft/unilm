# UniLM AI
<!--**Pre-trained models for natural language understanding (NLU) and generation (NLG) tasks**-->
**Pre-trained (foundation) models across tasks (understanding, generation and translation), languages (100+ languages), and modalities (language, image, audio, vision + language, audio + language, etc.)**

The family of UniLM AI:
> [**UniLM**](https://github.com/microsoft/unilm/tree/master/unilm) (```v1@NeurIPS'19 | v2@ICML'20 | v3@ACL'21```): **unified pre-training for language understanding and generation**

> [**InfoXLM**](https://github.com/microsoft/unilm/tree/master/infoxlm) (```v1@NAACL'21 | v2@ACL'21```): **multilingual/cross-lingual pre-trained models for 100+ languages**

> [**DeltaLM**](https://github.com/microsoft/unilm/tree/master/deltalm) (```NEW```): **encoder-decoder pre-training for language generation and translation for 100+ languages**

> [**MiniLM**](https://github.com/microsoft/unilm/tree/master/minilm) (```v1@NeurIPS'20 | v2@ACL'21```): **small and fast pre-trained models for language understanding and generation**

> [**AdaLM**](https://github.com/microsoft/unilm/tree/master/adalm) (```v1@ACL'21```): **domain, language, and task adaptation of pre-trained models**

> [**LayoutLM**](https://github.com/microsoft/unilm/tree/master/layoutlm) (```v1@KDD'20 | v2@ACL'21```): **multimodal (text + layout/format + image) pre-training for [Document AI](https://www.microsoft.com/en-us/research/project/document-ai/)** (e.g. scanned documents, PDF, etc.)

> [**LayoutXLM**](https://github.com/microsoft/unilm/tree/master/layoutxlm) (```NEW```): **multimodal (text + layout/format + image) pre-training for multilingual document understanding**

> [**LayoutReader**](https://github.com/microsoft/unilm/tree/master/layoutreader) (```EMNLP'21```): **Pre-training of text and layout for reading order detection**

> [**BEiT**](https://github.com/microsoft/unilm/tree/master/beit) (```NEW```): **BERT Pre-Training of Image Transformers**

> [**UniSpeech**](https://arxiv.org/abs/2101.07597) (```v1@ICML'21```): **Speech Pre-Training for ASR and TTS**

> [**s2s-ft**](https://github.com/microsoft/unilm/tree/master/s2s-ft): **sequence-to-sequence fine-tuning toolkit**

> [**XLM-T**](https://github.com/microsoft/unilm/tree/master/xlmt) (```NEW```): **Multilingual NMT w/ pretrained cross-lingual encoders**

> [**TrOCR**](https://github.com/microsoft/unilm/tree/master/trocr) (```NEW```): **Transformer-based OCR w/ pre-trained models**

## News
- [Model Release] September, 2021: [**TrOCR**](https://github.com/microsoft/unilm/tree/master/trocr) - Transformer-based OCR w/ pre-trained BEiT and RoBERTa models.
- August 2021: [**LayoutLMv2**](https://huggingface.co/transformers/master/model_doc/layoutlmv2.html) and [**LayoutXLM**](https://huggingface.co/transformers/master/model_doc/layoutxlm.html) are on [HuggingFace](https://github.com/huggingface/transformers)
- [Model Release] August, 2021: [**LayoutReader**](https://github.com/microsoft/unilm/tree/master/layoutreader) - Built with LayoutLM to improve general reading order detection.
- [Model Release] August, 2021: [**DeltaLM**](https://github.com/microsoft/unilm/tree/master/deltalm) - Encoder-decoder pre-training for language generation and translation.
- August 2021: [**BEiT**](https://huggingface.co/transformers/master/model_doc/beit.html) is on [HuggingFace](https://github.com/huggingface/transformers)
- [Model Release] July, 2021: [**BEiT**](https://github.com/microsoft/unilm/tree/master/beit) - Towards BERT moment for CV
- [Model Release] June, 2021: [**LayoutLMv2**](https://github.com/microsoft/unilm/tree/master/layoutlmv2), [**LayoutXLM**](https://github.com/microsoft/unilm/tree/master/layoutxlm), [**MiniLMv2**](https://github.com/microsoft/unilm/tree/master/minilm), and [**AdaLM**](https://github.com/microsoft/unilm/tree/master/adalm).
- May, 2021: [LayoutLMv2](https://github.com/microsoft/unilm/tree/master/layoutlmv2), InfoXLMv2, MiniLMv2, UniLMv3, and AdaLM were accepted by ACL 2021.
- April, 2021: [LayoutXLM](https://github.com/microsoft/unilm/tree/master/layoutxlm) is coming by extending the LayoutLM into multilingual support! A multilingual form understanding benchmark [XFUND](https://github.com/doc-analysis/XFUND) is also introduced, which includes forms with human labeled key-value pairs in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese).
- March, 2021: [InfoXLM](https://github.com/microsoft/unilm/tree/master/infoxlm) was accepted by NAACL 2021.
- December 29th, 2020: [LayoutLMv2](https://arxiv.org/abs/2012.14740) is coming with the new SOTA on a wide varierty of document AI tasks, including [DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1) and [SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) leaderboard.
- October 8th, 2020: T-ULRv2 (aka [InfoXLM](https://arxiv.org/abs/2007.07834)) as the SOTA on the [XTREME](https://sites.research.google/xtreme) leaderboard. // [Blog](https://www.microsoft.com/en-us/research/blog/microsoft-turing-universal-language-representation-model-t-ulrv2-tops-xtreme-leaderboard/)
- September, 2020: [MiniLM](https://github.com/microsoft/unilm/tree/master/minilm) was accepted by NeurIPS 2020.
- July 16, 2020: [**InfoXLM** (Multilingual UniLM)](https://github.com/microsoft/unilm/tree/master/infoxlm) [arXiv](https://arxiv.org/pdf/2007.07834.pdf)
- June, 2020: [UniLMv2](https://github.com/microsoft/unilm/tree/master/unilm) was accepted by ICML 2020; [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) was accepted by KDD 2020.
- April 5, 2020: [**Multilingual MiniLM**](https://github.com/microsoft/unilm/tree/master/minilm) released!
- September, 2019: [UniLMv1](https://github.com/microsoft/unilm/tree/master/unilm-v1) was accepted by NeurIPS 2019.

## Release

**\*\*\*\*\* ```New September, 2021```: [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) release \*\*\*\*\***

- [x] [**TrOCR**](https://github.com/microsoft/unilm/tree/master/trocr) (September 22, 2021): Transformer-based OCR with pre-trained models, which leverages the Transformer architecture for both image understanding and bpe-level text generation. The TrOCR model is simple but effective (convolution free), and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. "[TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282)"

**\*\*\*\*\* ```August, 2021```: [LayoutReader](https://github.com/microsoft/unilm/tree/master/layoutreader) release \*\*\*\*\***

- [x] [**LayoutReader**](https://github.com/microsoft/unilm/tree/master/layoutreader) (August 26, 2021): pre-training of text and layout for reading order detection. The pre-trained LayoutReader significantly improves both open-source and commercial OCR engines in ordering text lines. Meanwhile, we also created a reading order benchmark dataset [ReadingBank](https://github.com/doc-analysis/ReadingBank) to further empower the research in this area. "[LayoutReader: Pre-training of Text and Layout for Reading Order Detection](https://arxiv.org/abs/2108.11591) ```EMNLP 2021```"

**\*\*\*\*\* ```August, 2021```: [DeltaLM](https://github.com/microsoft/unilm/tree/master/deltalm) release \*\*\*\*\***

- [x] [**DeltaLM**](https://github.com/microsoft/unilm/tree/master/deltalm) (August, 2021): encoder-decoder pre-training for language generation and translation. DeltaLM **ranks first** on the [WMT21 multilingual translation task](http://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html). The task requires a model to translate between 102 languages. "[DeltaLM: Encoder-Decoder Pre-training for Language Generation and Translation by Augmenting Pretrained Multilingual Encoders.](https://arxiv.org/abs/2106.13736)"

**\*\*\*\*\* ```July, 2021```: [BEiT](https://github.com/microsoft/unilm/tree/master/beit) release \*\*\*\*\***

- [x] [**BEiT**](https://github.com/microsoft/unilm/tree/master/beit) (June 15, 2021): BERT Pre-Training of Image Transformers. BEiT-large achieves **[state-of-the-art results on ADE20K](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k) (a big jump to 57.0 mIoU) for semantic segmentation**. BEiT-large achieves **state-of-the-art ImageNet top-1 accuracy (88.6%) under the setting without extra data other than ImageNet-22k**. "[BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)"



**\*\*\*\*\* ```June, 2021```: [LayoutXLM](https://github.com/microsoft/unilm/tree/master/layoutxlm) | [AdaLM](https://github.com/microsoft/unilm/tree/master/adalm) | [MiniLMv2](https://github.com/microsoft/unilm/tree/master/minilm) release \*\*\*\*\***

- [x] [**LayoutXLM**](https://github.com/microsoft/unilm/tree/master/layoutxlm) (April 17, 2021): multimodal pre-training for multilingual visually-rich document understanding. The pre-trained LayoutXLM model has significantly outperformed the existing SOTA cross-lingual pre-trained models on the FUNSD and multilingual [XFUND](https://github.com/doc-analysis/XFUND) dataset including 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese). "[LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836)"
- [x] [**AdaLM**](https://github.com/microsoft/unilm/tree/master/adalm) (June 2021): a simple yet effective approach for domain adaptation of pre-trained models. Biomedical specific pre-trained models are released. "[Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains](#) ```ACL 2021```"
- [x] [**MiniLMv2**](https://github.com/microsoft/unilm/tree/master/minilm) (December, 2020): a simple yet effective task-agnostic knoweldge distillation method, namely multi-head self-attention relation distillation, for compressing large pre-trained Transformers into small and fast pre-trained models. MiniLMv2 significantly outperforms MiniLMv1. Both English and multilingual MiniLM models are released. "[MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers](https://arxiv.org/abs/2012.15828) ```ACL 2021```"

**\*\*\*\*\* ```May, 2021```: [LayoutLMv2](https://github.com/microsoft/unilm/tree/master/layoutlmv2) | [LayoutXLM](https://github.com/microsoft/unilm/tree/master/layoutxlm) release \*\*\*\*\***

- [x] [**LayoutLM 2.0**](https://github.com/microsoft/unilm/tree/master/layoutlmv2) (December 29, 2020): multimodal pre-training for visually-rich document understanding by leveraging text, layout and image information in a single framework. It is coming with new SOTA on a wide range of document understanding tasks, including FUNSD (0.7895 -> 0.8420), CORD (0.9493 -> 0.9601), SROIE (0.9524 -> 0.9781), Kleister-NDA (0.834 -> 0.852), RVL-CDIP (0.9443 -> 0.9564), and DocVQA (0.7295 -> 0.8672). "[LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740) ```ACL 2021```"

**\*\*\*\*\* ```February, 2020```: [UniLM v2](https://github.com/microsoft/unilm/tree/master/unilm) | [MiniLM v1](https://github.com/microsoft/unilm/tree/master/minilm) | [LayoutLM v1](https://github.com/microsoft/unilm/tree/master/layoutlm) | [s2s-ft v1](https://github.com/microsoft/unilm/tree/master/s2s-ft) release \*\*\*\*\***

- [x] [**LayoutLM 1.0**](https://github.com/microsoft/unilm/tree/master/layoutlm) (February 18, 2020): pre-trained models for document (image) understanding (e.g. receipts, forms, etc.) . It achieves new SOTA results in several downstream tasks, including form understanding (the FUNSD dataset from 70.72 to 79.27), receipt understanding (the [ICDAR 2019 SROIE leaderboard](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) from 94.02 to 95.24) and document image classification (the RVL-CDIP dataset from 93.07 to 94.42). "[LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) ```KDD 2020```"
- [x] [**s2s-ft 1.0**](https://github.com/microsoft/unilm/tree/master/s2s-ft) (February 26, 2020): A PyTorch package used to fine-tune pre-trained Transformers for sequence-to-sequence language generation. "[s2s-ft: Fine-Tuning Pre-Trained Transformers for Sequence-to-Sequence Learning](#)"
- [x] [**MiniLM 1.0**](https://github.com/microsoft/unilm/tree/master/minilm) (February 26, 2020): deep self-attention distillation is all you need (for task-agnostic knowledge distillation of pre-trained Transformers). MiniLM (12-layer, 384-hidden) achieves 2.7x speedup and comparable results over BERT-base (12-layer, 768-hidden) on NLU tasks as well as strong results on NLG tasks. The even smaller MiniLM (6-layer, 384-hidden) obtains 5.3x speedup and produces very competitive results. "[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957) ```NeurIPS 2020```"
- [x] [**UniLM 2.0**](https://github.com/microsoft/unilm/tree/master/unilm) (February 28, 2020): **unified pre-training** of bi-directional LM (via autoencoding) and sequence-to-sequence LM (via partially autoregressive) w/ **Pseudo-Masked Language Model** for language understanding and generation. UniLM v2 achieves new SOTA in a wide range of natural language understanding and generation tasks. "[UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training](https://arxiv.org/abs/2002.12804) ```ICML 2020```"



**\*\*\*\*\* October 1st, 2019: UniLM v1 release \*\*\*\*\***

- [x] [**UniLM v1**](https://github.com/microsoft/unilm/tree/master/unilm-v1) (September 30, 2019): the code and pre-trained models for the ```NeurIPS 2019``` paper entitled "[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)". UniLM (v1) achieves the **new SOTA results** in **NLG** (especially **sequence-to-sequence generation**) tasks, including abstractive summarization (the Gigaword and CNN/DM datasets), question generation (the SQuAD QG dataset), etc. 

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using UniLM AI models, please submit a GitHub issue.

For other communications related to UniLM AI, please contact Li Dong (`lidong1@microsoft.com`), [Furu Wei](http://gitnlp.org/) (`fuwei@microsoft.com`).

