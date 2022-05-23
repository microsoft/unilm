# Aggressive Decoding

**Lossless Acceleration for Seq2seq Generation with Aggressive Decoding. https://arxiv.org/pdf/2205.10350.pdf**

- May 2022: preprint [arXiv](https://arxiv.org/pdf/2205.10350.pdf) released; code updated and integrated

## Introduction

Aggressive Decoding, a novel decoding paradigm for **lossless** speedup of seq2seq generation. Unlike the previous efforts (e.g., non-autoregressive decoding) speeding up seq2seq generation at the cost of quality loss, Aggressive Decoding aims to yield the identical (or better) generation compared with autoregressive decoding but in a significant speedup: For the seq2seq tasks characterized by highly similar inputs and outputs (e.g., Grammatical Error Correction and Text Simplification), the **Input-guided Aggressive Decoding (IAD)** can introduce a **7x-9x** speedup for the popular 6-layer Transformer on GPU with the **identical results as greedy decoding**; For other general seq2seq tasks (e.g., Machine Translation and Abstractive Summarization), the **Generalized Aggressive Decoding (GAD)** can have a **3x-5x** speedup with the **identical or even better quality**.

Please check out IAD and GAD in the sub-folders.

## Acknowledgement

This repository is built using the [Fairseq](https://github.com/pytorch/fairseq) repository.

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For other communications related to Aggressive Decoding, please contact [Tao Ge](https://www.microsoft.com/en-us/research/people/tage/) (`tage@microsoft.com`).
