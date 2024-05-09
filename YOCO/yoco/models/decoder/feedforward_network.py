import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.model_parallel.megatron.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from .kernel.swiglu import swiglu
from .model_parallel_init import init_method

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        load_checkpoint=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = ColumnParallelLinear(self.embed_dim, ffn_dim, bias=False, gather_output=False, init_method=init_method)
        self.gate = ColumnParallelLinear(self.embed_dim, ffn_dim, bias=False, gather_output=False, init_method=init_method)
        self.fc2 = RowParallelLinear(ffn_dim, self.embed_dim, bias=False, input_is_parallel=True, init_method=init_method)

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc2(swiglu(self.fc1(x), self.gate(x)))
        output = x.view(x_shape)
        return output