import attr
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logit_laplace_eps: float = 0.1

@attr.s(eq=False)
class Conv2d(nn.Module):
	n_in:  int = attr.ib(validator=lambda i, a, x: x >= 1)
	n_out: int = attr.ib(validator=lambda i, a, x: x >= 1)
	kw:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 2 == 1)

	use_float16:   bool         = attr.ib(default=True)
	device:        torch.device = attr.ib(default=torch.device('cpu'))
	requires_grad: bool         = attr.ib(default=False)

	def __attrs_post_init__(self) -> None:
		super().__init__()

		w = torch.empty((self.n_out, self.n_in, self.kw, self.kw), dtype=torch.float32,
			device=self.device, requires_grad=self.requires_grad)
		w.normal_(std=1 / math.sqrt(self.n_in * self.kw ** 2))

		b = torch.zeros((self.n_out,), dtype=torch.float32, device=self.device,
			requires_grad=self.requires_grad)
		self.w, self.b = nn.Parameter(w), nn.Parameter(b)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.use_float16 and 'cuda' in self.w.device.type:
			if x.dtype != torch.float16:
				x = x.half()

			w, b = self.w.half(), self.b.half()
		else:
			if x.dtype != torch.float32:
				x = x.float()

			w, b = self.w, self.b

		return F.conv2d(x, w, b, padding=(self.kw - 1) // 2)

def map_pixels(x: torch.Tensor) -> torch.Tensor:
	if x.dtype != torch.float:
		raise ValueError('expected input to have type float')

	return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def unmap_pixels(x: torch.Tensor) -> torch.Tensor:
	if len(x.shape) != 4:
		raise ValueError('expected input to be 4d')
	if x.dtype != torch.float:
		raise ValueError('expected input to have type float')

	return torch.clamp((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)
