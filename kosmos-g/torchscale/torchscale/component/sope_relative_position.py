# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
from scipy.optimize import minimize

import torch
import torch.nn as nn

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


class SoPE(nn.Module):
    def __init__(
        self, head_dim, scale_base = 512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, len, offset=0):
        min = -(len + offset) // 2
        max = len + offset + min
        scale = self.scale ** torch.arange(min, max, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)
        return (sin, cos, scale)