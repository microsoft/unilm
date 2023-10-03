# TorchScale - A Library for Transformers at (Any) Scale

<p>
  <a href="https://github.com/microsoft/torchscale/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/torchscale"><img alt="MIT License" src="https://badge.fury.io/py/torchscale.svg" /></a>
</p>

TorchScale is a PyTorch library that allows researchers and developers to scale up Transformers efficiently and effectively.
It has the implementation of fundamental research to improve modeling generality and capability, as well as training stability and efficiency of scaling Transformers.

- Stability - [**DeepNet**](https://arxiv.org/abs/2203.00555): scaling Transformers to 1,000 Layers and beyond
- Generality - [**Foundation Transformers (Magneto)**](https://arxiv.org/abs/2210.06423): towards true general-purpose modeling across tasks and modalities (including language, vision, speech, and multimodal)
- Efficiency - [**X-MoE**](https://arxiv.org/abs/2204.09179): scalable & finetunable sparse Mixture-of-Experts (MoE)

## News

- November, 2022: TorchScale 0.1.1 released [[Paper](https://arxiv.org/abs/2211.13184)] [[PyPI](https://pypi.org/project/torchscale/)]

## Installation

To install:
```
pip install torchscale
```

Alternatively, you can develop it locally:
```
git clone https://github.com/microsoft/torchscale.git
cd torchscale
pip install -e .
```

## Getting Started

It takes only several lines of code to create a model with the above fundamental research features enabled. Here is how to quickly obtain a BERT-like encoder:

```python
>>> from torchscale.architecture.config import EncoderConfig
>>> from torchscale.architecture.encoder import Encoder

>>> config = EncoderConfig(vocab_size=64000)
>>> model = Encoder(config)

>>> print(model)
```

We also support the `Decoder` architecture and the `EncoderDecoder` architecture:

```python
# Creating a decoder model
>>> from torchscale.architecture.config import DecoderConfig
>>> from torchscale.architecture.decoder import Decoder

>>> config = DecoderConfig(vocab_size=64000)
>>> decoder = Decoder(config)
>>> print(decoder)

# Creating a encoder-decoder model
>>> from torchscale.architecture.config import EncoderDecoderConfig
>>> from torchscale.architecture.encoder_decoder import EncoderDecoder

>>> config = EncoderDecoderConfig(vocab_size=64000)
>>> encdec = EncoderDecoder(config)
>>> print(encdec)
```

## Key Features

- [DeepNorm to improve the training stability of Post-LayerNorm Transformers](https://arxiv.org/abs/2203.00555)
  * enabled by setting *deepnorm=True* in the `Config` class. 
  * It adjusts both the residual connection and the initialization method according to the model architecture (i.e., encoder, decoder, or encoder-decoder).

- [SubLN for the model generality and the training stability](https://arxiv.org/abs/2210.06423)
  * enabled by *subln=True*. This is enabled by default. 
  * It introduces another LayerNorm to each sublayer and adjusts the initialization according to the model architecture.
  * Note that SubLN and DeepNorm cannot be used in one single model.

- [X-MoE: efficient and finetunable sparse MoE modeling](https://arxiv.org/abs/2204.09179)
  * enabled by *use_xmoe=True*. 
  * It replaces every *'moe_freq'* `FeedForwardNetwork` layers with the X-MoE layers.

- [Multiway architecture for multimodality](https://arxiv.org/abs/2208.10442)
  * enabled by *multiway=True*.
  * It provides a pool of Transformer's parameters used for different modalities.

- [Relative position bias](https://arxiv.org/abs/1910.10683)
  * enabled by adjusting *rel_pos_buckets* and *max_rel_pos*.

- [SparseClip: improving the gradient clipping for sparse MoE models](https://arxiv.org/abs/2211.13184)
  * we provide a [sample code](examples/fairseq/utils/sparse_clip.py) that can be easily adapted to the FairSeq (or other) repo.

Most of the features above can be used by simply passing the corresponding parameters to the config. For example:

```python
>>> from torchscale.architecture.config import EncoderConfig
>>> from torchscale.architecture.encoder import Encoder

>>> config = EncoderConfig(vocab_size=64000, deepnorm=True, multiway=True)
>>> model = Encoder(config)

>>> print(model)
```

## Examples

We have the examples of how to use TorchScale in the following scenarios/tasks:

- Language

  * [Decoder/GPT](examples/fairseq/README.md#example-gpt-pretraining)

  * [Encoder-Decoder/Neural Machine Translation](examples/fairseq/README.md#example-machine-translation)

  * [Encoder/BERT](examples/fairseq/README.md#example-bert-pretraining)

- Vision

  * ViT/BEiT [In progress]

- Speech

- Multimodal

  * [Multiway Transformers/BEiT-3](torchscale/model/BEiT3.py) [In progress]

We plan to provide more examples regarding different tasks (e.g. vision pretraining and speech recognition) and various deep learning toolkits (e.g. [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)). Any comments or PRs are welcome!

## Results

### Stability Evaluation

<p align="center">
  <img src="https://publicmodel.blob.core.windows.net/torchscale/pic/convergence.png" width="800"/>
</p>

The training curve is smooth by using TorchScale, while the baseline Transformer cannot converge.

### Scaling-up Experiments

<p align="center">
  <img src="https://publicmodel.blob.core.windows.net/torchscale/pic/scaling_curve.png" width="800"/>
</p>

TorchScale supports arbitrary depths and widths, successfully scaling-up the models without pain.

## Acknowledgments

Some implementations in TorchScale are either adapted from or inspired by the [FairSeq](https://github.com/facebookresearch/fairseq) repository and the [UniLM](https://github.com/microsoft/unilm) repository.

## Citations

If you find this repository useful, please consider citing our work:

```
@article{torchscale,
  author    = {Shuming Ma and Hongyu Wang and Shaohan Huang and Wenhui Wang and Zewen Chi and Li Dong and Alon Benhaim and Barun Patra and Vishrav Chaudhary and Xia Song and Furu Wei},
  title     = {{TorchScale}: {Transformers} at Scale},
  journal   = {CoRR},
  volume    = {abs/2211.13184},
  year      = {2022}
}
```

```
@article{deepnet,
  author    = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Dongdong Zhang and Furu Wei},
  title     = {{DeepNet}: Scaling {Transformers} to 1,000 Layers},
  journal   = {CoRR},
  volume    = {abs/2203.00555},
  year      = {2022},
}
```

```
@article{magneto,
  author    = {Hongyu Wang and Shuming Ma and Shaohan Huang and Li Dong and Wenhui Wang and Zhiliang Peng and Yu Wu and Payal Bajaj and Saksham Singhal and Alon Benhaim and Barun Patra and Zhun Liu and Vishrav Chaudhary and Xia Song and Furu Wei},
  title     = {Foundation {Transformers}},
  journal   = {CoRR},
  volume    = {abs/2210.06423},
  year      = {2022}
}
```

```
@inproceedings{xmoe,
  title={On the Representation Collapse of Sparse Mixture of Experts},
  author={Zewen Chi and Li Dong and Shaohan Huang and Damai Dai and Shuming Ma and Barun Patra and Saksham Singhal and Payal Bajaj and Xia Song and Xian-Ling Mao and Heyan Huang and Furu Wei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=mWaYC6CZf5}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [Furu Wei](mailto:fuwei@microsoft.com) and [Shuming Ma](mailto:shumma@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
