import math

import torch
import torch.nn as nn

def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

def qkvg_init_method(tensor, **kwargs):
    nn.init.xavier_uniform_(tensor, gain = 2 ** -2.5)

def out_init_method(tensor, **kwargs):
    nn.init.xavier_uniform_(tensor, gain = 2 ** -1)

def vocab_init_method(tensor, **kwargs):
    torch.nn.init.normal_(tensor, mean=0, std=tensor.shape[1] ** -0.5)
