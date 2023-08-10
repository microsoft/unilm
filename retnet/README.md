# Retentive Network: The Successor to Transformer for Large Language Models

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

- Aug 4, 2023: fix a bug of the chunkwise recurrent representation ([commit](https://github.com/microsoft/torchscale/commit/0b1f113985a0339bc322b0c7df91be0f745cb311))
- Aug 4, 2023: improve the numerical precision of the recurrent representation as suggested by https://github.com/microsoft/torchscale/issues/47 ([commit](https://github.com/microsoft/torchscale/commit/7f0bf80a7e41e6fe2d3bf1fda570fbbf8ecc13a4))
- Aug 2, 2023: update eps of LN to 1e-6 ([commit](https://github.com/microsoft/torchscale/commit/2c29de0fb3e5e559181f0fb4854330c5b35961cd))

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
