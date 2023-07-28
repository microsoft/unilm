# Retentive Network: The Successor to Transformer for Large Language Models

- Code release: [https://github.com/microsoft/torchscale](https://github.com/microsoft/torchscale/commit/bf65397b26469ac9c24d83a9b779b285c1ec640b)
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
