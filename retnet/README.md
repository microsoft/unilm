# Retentive Network: The Successor to Transformer for Large Language Models

- May 2024: Gated RetNet (i.e., RetNet-3) as part of YOCO / [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/abs/2405.05254)
- Code release: [https://github.com/microsoft/torchscale](https://github.com/microsoft/torchscale)
- July 2023: release preprint [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)

<p>
  <a href="https://github.com/microsoft/torchscale/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/torchscale"><img alt="MIT License" src="https://badge.fury.io/py/torchscale.svg" /></a>
</p>

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

It takes only several lines of code to create a RetNet model:

```python
# Creating a RetNet model
>>> import torch
>>> from torchscale.architecture.config import RetNetConfig
>>> from torchscale.architecture.retnet import RetNetDecoder

>>> config = RetNetConfig(vocab_size=64000)
>>> retnet = RetNetDecoder(config)

>>> print(retnet)
```

## Changelog

- Nov 2023: improve stability via [better initialization](https://github.com/microsoft/torchscale/commit/ff7c1f286503a4aa84ee90dbd68ee4d5af465d12)
- Nov 2023: fix retention normalization in the [commit](https://github.com/microsoft/torchscale/commit/fdd8838a756c7c435d7f8a1e4303e150dfac7442)
- Oct 2023: improve stability as follows
  - The RMSNorm is used in the [commit](https://github.com/microsoft/torchscale/commit/5c89ffbeea3ba458a865a569f947bf82cca50090), so that the effects of LN_eps can be eliminated
  - The LN eps was modified from 1e-6 to 1e-5 as in the [commit](https://github.com/microsoft/torchscale/commit/d1fefe9c22bad07535f56c4c461b94588dd8cc84)
  - For the RetNet implementation, the initialization principle proposed in DeepNet has been integrated. So the arguments `--subln or --deepnorm` should not be added.
  - Removing layer bias also improves training stability
- Aug 4, 2023: fix a bug of the chunkwise recurrent representation ([commit](https://github.com/microsoft/torchscale/commit/0b1f113985a0339bc322b0c7df91be0f745cb311))
- Aug 4, 2023: improve the numerical precision of the recurrent representation as suggested by https://github.com/microsoft/torchscale/issues/47 ([commit](https://github.com/microsoft/torchscale/commit/7f0bf80a7e41e6fe2d3bf1fda570fbbf8ecc13a4))

## Citations

If you find this repository useful, please consider citing our work:

```
@article{retnet,
  author={Yutao Sun and Li Dong and Shaohan Huang and Shuming Ma and Yuqing Xia and Jilong Xue and Jianyong Wang and Furu Wei},
  title     = {Retentive Network: A Successor to {Transformer} for Large Language Models},
  journal   = {ArXiv},
  volume    = {abs/2307.08621},
  year      = {2023}
}
```
