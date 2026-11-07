"""DALL-E dVAE image tokenizer (8192 visual tokens), as used for BEiT v1's MIM targets.

The encoder architecture below re-implements OpenAI's ``dall_e`` encoder
(https://github.com/openai/DALL-E, modified MIT licence, notice below) with identical parameter names,
so the public ``encoder.pkl`` weights load into it. We do not depend on the
``dall_e`` package itself: ``encoder.pkl`` is a pickled ``dall_e.encoder.Encoder``
object, and ``load_dalle_encoder`` unpickles it with a restricted unpickler that
maps only the expected classes onto plain stand-ins, then copies the weights.

As in BEiT, the tokenizer sees a 112x112 view of the 224x224 image, and its 8x
down-sampling gives a 14x14 token grid, one token per LayoutLMv3 16x16 patch.

Note: this dVAE was trained by OpenAI on natural images, not documents. The
LayoutLMv3 paper used a document-trained dVAE from DiT, whose weights are not
released with LayoutLMv3.

The encoder architecture is adapted from openai/DALL-E, distributed under this
licence notice:

    Modified MIT License
    Software Copyright (c) 2021 OpenAI

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions: The above copyright
    notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import io
import pickle
import warnings
import zipfile
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DALLE_ENCODER_URL = "https://cdn.openai.com/dall-e/encoder.pkl"
DALLE_DECODER_URL = "https://cdn.openai.com/dall-e/decoder.pkl"
LOGIT_LAPLACE_EPS = 0.1


def map_pixels(x: torch.Tensor) -> torch.Tensor:
    """dall_e.utils.map_pixels: [0, 1] -> [eps, 1 - eps]."""
    return (1 - 2 * LOGIT_LAPLACE_EPS) * x + LOGIT_LAPLACE_EPS


def unmap_pixels(x: torch.Tensor) -> torch.Tensor:
    """dall_e.utils.unmap_pixels: [eps, 1 - eps] -> [0, 1]."""
    return torch.clamp((x - LOGIT_LAPLACE_EPS) / (1 - 2 * LOGIT_LAPLACE_EPS), 0, 1)


class Conv2d(nn.Module):
    """dall_e.utils.Conv2d: same-padding convolution with parameters named ``w`` and ``b``."""

    def __init__(self, n_in: int, n_out: int, kw: int):
        super().__init__()
        self.kw = kw
        self.w = nn.Parameter(torch.zeros(n_out, n_in, kw, kw))
        self.b = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        return F.conv2d(x, self.w, self.b, padding=(self.kw - 1) // 2)


class EncoderBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_layers: int):
        super().__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / (n_layers ** 2)
        self.id_path = Conv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
            ("relu_1", nn.ReLU()), ("conv_1", Conv2d(n_in, n_hid, 3)),
            ("relu_2", nn.ReLU()), ("conv_2", Conv2d(n_hid, n_hid, 3)),
            ("relu_3", nn.ReLU()), ("conv_3", Conv2d(n_hid, n_hid, 3)),
            ("relu_4", nn.ReLU()), ("conv_4", Conv2d(n_hid, n_out, 1))]))

    def forward(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)


class DalleEncoder(nn.Module):
    """dall_e.encoder.Encoder with its default sizes: (B, 3, H, W) in [eps, 1-eps] -> (B, 8192, H/8, W/8) logits."""

    def __init__(self, group_count: int = 4, n_hid: int = 256, n_blk_per_group: int = 2,
                 input_channels: int = 3, vocab_size: int = 8192):
        super().__init__()
        self.vocab_size = vocab_size
        n_layers = group_count * n_blk_per_group

        def group(n_in, n_out, pool):
            blocks = [(f"block_{i + 1}", EncoderBlock(n_in if i == 0 else n_out, n_out, n_layers))
                      for i in range(n_blk_per_group)]
            if pool:
                blocks.append(("pool", nn.MaxPool2d(kernel_size=2)))
            return nn.Sequential(OrderedDict(blocks))

        self.blocks = nn.Sequential(OrderedDict([
            ("input", Conv2d(input_channels, n_hid, 7)),
            ("group_1", group(n_hid, n_hid, True)),
            ("group_2", group(n_hid, 2 * n_hid, True)),
            ("group_3", group(2 * n_hid, 4 * n_hid, True)),
            ("group_4", group(4 * n_hid, 8 * n_hid, False)),
            ("output", nn.Sequential(OrderedDict([("relu", nn.ReLU()),
                                                  ("conv", Conv2d(8 * n_hid, vocab_size, 1))]))),
        ]))

    def forward(self, x):
        return self.blocks(x.float())


class DecoderBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_layers: int):
        super().__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / (n_layers ** 2)
        self.id_path = Conv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
            ("relu_1", nn.ReLU()), ("conv_1", Conv2d(n_in, n_hid, 1)),
            ("relu_2", nn.ReLU()), ("conv_2", Conv2d(n_hid, n_hid, 3)),
            ("relu_3", nn.ReLU()), ("conv_3", Conv2d(n_hid, n_hid, 3)),
            ("relu_4", nn.ReLU()), ("conv_4", Conv2d(n_hid, n_out, 3))]))

    def forward(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)


class DalleDecoder(nn.Module):
    """dall_e.decoder.Decoder with its default sizes: (B, 8192, h, w) one-hot codes -> (B, 6, 8h, 8w).

    Channels 0-2 are the logit-Laplace means; ``unmap_pixels(sigmoid(out[:, :3]))`` is the image in [0, 1].
    """

    def __init__(self, group_count: int = 4, n_init: int = 128, n_hid: int = 256, n_blk_per_group: int = 2,
                 output_channels: int = 3, vocab_size: int = 8192):
        super().__init__()
        n_layers = group_count * n_blk_per_group

        def group(n_in, n_out, up):
            blocks = [(f"block_{i + 1}", DecoderBlock(n_in if i == 0 else n_out, n_out, n_layers))
                      for i in range(n_blk_per_group)]
            if up:
                blocks.append(("upsample", nn.Upsample(scale_factor=2, mode="nearest")))
            return nn.Sequential(OrderedDict(blocks))

        self.blocks = nn.Sequential(OrderedDict([
            ("input", Conv2d(vocab_size, n_init, 1)),
            ("group_1", group(n_init, 8 * n_hid, True)),
            ("group_2", group(8 * n_hid, 4 * n_hid, True)),
            ("group_3", group(4 * n_hid, 2 * n_hid, True)),
            ("group_4", group(2 * n_hid, n_hid, False)),
            ("output", nn.Sequential(OrderedDict([("relu", nn.ReLU()),
                                                  ("conv", Conv2d(n_hid, 2 * output_channels, 1))]))),
        ]))

    def forward(self, z):
        return self.blocks(z.float())

    def reconstruct(self, z):
        return unmap_pixels(torch.sigmoid(self(z)[:, :3]))


class _Stub(nn.Module):
    """Stand-in for dall_e classes while unpickling; only its parameters/submodules are used."""

    def __init__(self, *args, **kwargs):  # never called by pickle
        super().__init__()


class _RestrictedUnpickler(pickle.Unpickler):
    _ALLOWED = {
        ("collections", "OrderedDict"), ("builtins", "set"), ("__builtin__", "set"),
        ("torch", "device"), ("torch", "FloatStorage"), ("torch", "HalfStorage"),
        ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
        ("torch.nn.modules.activation", "ReLU"), ("torch.nn.modules.container", "Sequential"),
        ("torch.nn.modules.pooling", "MaxPool2d"), ("torch.nn.modules.linear", "Identity"),
        ("torch.nn.modules.upsampling", "Upsample"),
    }
    _DALLE = {("dall_e.encoder", "Encoder"), ("dall_e.encoder", "EncoderBlock"), ("dall_e.utils", "Conv2d"),
              ("dall_e.decoder", "Decoder"), ("dall_e.decoder", "DecoderBlock")}

    def __init__(self, file, zf: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if (module, name) in self._DALLE:
            return _Stub
        if (module, name) in self._ALLOWED:
            module = "builtins" if module == "__builtin__" else module
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"unexpected global in dVAE checkpoint: {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel) -- torch's zip serialization format
        _, storage_type, key, _, numel = pid
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        with warnings.catch_warnings():                # TypedStorage deprecation notices
            warnings.simplefilter("ignore", UserWarning)
            dtype = storage_type.dtype
            t = torch.frombuffer(bytearray(raw), dtype=dtype)[:numel]
            return torch.storage.TypedStorage(wrap_storage=t.untyped_storage(), dtype=dtype, _internal=True)


def _state_dict_from_pkl(path: str | Path) -> dict[str, torch.Tensor]:
    with zipfile.ZipFile(path) as zf:
        data_pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = data_pkl.rsplit("/", 1)[0]
        obj = _RestrictedUnpickler(io.BytesIO(zf.read(data_pkl)), zf, prefix).load()
    return {k: v.detach().clone().float() for k, v in obj.state_dict().items()}


def load_dalle_encoder(path: str | Path | None = None, cache_dir: str | Path | None = None) -> DalleEncoder:
    """Load OpenAI's DALL-E dVAE encoder from ``encoder.pkl`` (downloaded if ``path`` is None).

    A plain ``state_dict`` saved with ``torch.save`` (``.pt``) is also accepted.
    """
    if path is None:
        cache = Path(cache_dir or Path.home() / ".cache" / "layoutlmv3_pretraining")
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "dalle_encoder.pkl"
        if not path.exists():
            torch.hub.download_url_to_file(DALLE_ENCODER_URL, str(path))
    path = Path(path)
    if path.suffix == ".pt":
        sd = torch.load(path, map_location="cpu", weights_only=True)
    else:
        sd = _state_dict_from_pkl(path)
    enc = DalleEncoder()
    enc.load_state_dict(sd, strict=True)
    return enc.eval().requires_grad_(False)


def load_dalle_decoder(path: str | Path | None = None, cache_dir: str | Path | None = None) -> DalleDecoder:
    """Load OpenAI's DALL-E dVAE decoder from ``decoder.pkl`` (downloaded if ``path`` is None) or a ``.pt``."""
    if path is None:
        cache = Path(cache_dir or Path.home() / ".cache" / "layoutlmv3_pretraining")
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "dalle_decoder.pkl"
        if not path.exists():
            torch.hub.download_url_to_file(DALLE_DECODER_URL, str(path))
    path = Path(path)
    sd = torch.load(path, map_location="cpu", weights_only=True) if path.suffix == ".pt" else _state_dict_from_pkl(path)
    dec = DalleDecoder()
    dec.load_state_dict(sd, strict=True)
    return dec.eval().requires_grad_(False)


class DalleTokenizer:
    """MIM targets from the DALL-E dVAE: ``encode(pixel_values) -> LongTensor[B, 196]`` with 8192 tokens.

    ``pixel_values`` are LayoutLMv3Processor outputs (224x224, normalised with mean = std = 0.5).
    They are mapped back to [0, 1], resized to ``view_size`` (112) and passed through ``map_pixels``.
    """

    def __init__(self, encoder: DalleEncoder, view_size: int = 112, image_mean: float = 0.5,
                 image_std: float = 0.5):
        self.encoder = encoder
        self.view_size, self.mean, self.std = view_size, image_mean, image_std

    @classmethod
    def from_pretrained(cls, path: str | Path | None = None, **kwargs) -> "DalleTokenizer":
        return cls(load_dalle_encoder(path), **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.encoder.vocab_size

    def to(self, device) -> "DalleTokenizer":
        self.encoder.to(device)
        return self

    def preprocess(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = (pixel_values.float() * self.std + self.mean).clamp(0, 1)
        if x.shape[-1] != self.view_size or x.shape[-2] != self.view_size:
            x = F.interpolate(x, size=(self.view_size, self.view_size), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        return map_pixels(x)

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        dev = next(self.encoder.parameters()).device
        logits = self.encoder(self.preprocess(pixel_values.to(dev)))       # B, 8192, 14, 14
        return logits.argmax(1).flatten(1).to(pixel_values.device)         # row-major, like the patches
