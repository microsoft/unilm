import attr
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections  import OrderedDict
from functools    import partial
from dall_e.utils import Conv2d

@attr.s(eq=False, repr=False)
class EncoderBlock(nn.Module):
	n_in:     int = attr.ib(validator=lambda i, a, x: x >= 1)
	n_out:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 ==0)
	n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

	device:        torch.device = attr.ib(default=None)
	requires_grad: bool         = attr.ib(default=False)

	def __attrs_post_init__(self) -> None:
		super().__init__()
		self.n_hid = self.n_out // 4
		self.post_gain = 1 / (self.n_layers ** 2)

		make_conv     = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
		self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
		self.res_path = nn.Sequential(OrderedDict([
				('relu_1', nn.ReLU()),
				('conv_1', make_conv(self.n_in,  self.n_hid, 3)),
				('relu_2', nn.ReLU()),
				('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
				('relu_3', nn.ReLU()),
				('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
				('relu_4', nn.ReLU()),
				('conv_4', make_conv(self.n_hid, self.n_out, 1)),]))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.id_path(x) + self.post_gain * self.res_path(x)

@attr.s(eq=False, repr=False)
class Encoder(nn.Module):
	group_count:     int = 4
	n_hid:           int = attr.ib(default=256,  validator=lambda i, a, x: x >= 64)
	n_blk_per_group: int = attr.ib(default=2,    validator=lambda i, a, x: x >= 1)
	input_channels:  int = attr.ib(default=3,    validator=lambda i, a, x: x >= 1)
	vocab_size:      int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

	device:              torch.device = attr.ib(default=torch.device('cpu'))
	requires_grad:       bool         = attr.ib(default=False)
	use_mixed_precision: bool         = attr.ib(default=True)

	def __attrs_post_init__(self) -> None:
		super().__init__()

		blk_range  = range(self.n_blk_per_group)
		n_layers   = self.group_count * self.n_blk_per_group
		make_conv  = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
		make_blk   = partial(EncoderBlock, n_layers=n_layers, device=self.device,
				requires_grad=self.requires_grad)

		self.blocks = nn.Sequential(OrderedDict([
			('input', make_conv(self.input_channels, 1 * self.n_hid, 7)),
			('group_1', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
				('pool', nn.MaxPool2d(kernel_size=2)),
			]))),
			('group_2', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(1 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
				('pool', nn.MaxPool2d(kernel_size=2)),
			]))),
			('group_3', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
				('pool', nn.MaxPool2d(kernel_size=2)),
			]))),
			('group_4', nn.Sequential(OrderedDict([
				*[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
			]))),
			('output', nn.Sequential(OrderedDict([
				('relu', nn.ReLU()),
				('conv', make_conv(8 * self.n_hid, self.vocab_size, 1, use_float16=False)),
			]))),
		]))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if len(x.shape) != 4:
			raise ValueError(f'input shape {x.shape} is not 4d')
		if x.shape[1] != self.input_channels:
			raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
		if x.dtype != torch.float32:
			raise ValueError('input must have dtype torch.float32')

		return self.blocks(x)
