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
