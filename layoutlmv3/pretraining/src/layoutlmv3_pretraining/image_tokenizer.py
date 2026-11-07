"""Discrete image tokenizers that produce MIM targets.

The LayoutLMv3 paper reconstructs discrete visual tokens produced by a dVAE
image tokenizer trained on documents (from DiT). That tokenizer's weights are
not distributed with the paper's release, so this module defines a small
interface and a dependency-free default:

* ``KMeansPatchTokenizer`` - a k-means codebook over down-sampled grey-scale
  patches, fitted on your own document images. It is far simpler than a dVAE,
  but it gives MIM a meaningful discrete target and needs no external weights.

Any object with ``vocab_size`` and ``encode(pixel_values) -> LongTensor[B, N]``
can be used instead; ``dvae.DalleTokenizer`` wraps OpenAI's DALL-E dVAE that way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class KMeansPatchTokenizer:
    def __init__(self, centroids: np.ndarray, patch_size: int = 16, pool: int = 4):
        self.centroids = torch.as_tensor(centroids, dtype=torch.float32)
        self.patch_size, self.pool = patch_size, pool

    @property
    def vocab_size(self) -> int:
        return int(self.centroids.shape[0])

    @staticmethod
    def features(pixel_values: torch.Tensor, patch_size: int = 16, pool: int = 4) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, pool*pool) grey-scale, average-pooled patch features."""
        grey = pixel_values.float().mean(dim=1, keepdim=True)                  # B,1,H,W
        small = F.avg_pool2d(grey, kernel_size=patch_size // pool)             # B,1,H/4,W/4
        b, _, h, w = small.shape
        g = patch_size // (patch_size // pool)                                 # = pool
        x = small.reshape(b, h // g, g, w // g, g).permute(0, 1, 3, 2, 4)      # B,gh,gw,g,g
        return x.reshape(b, (h // g) * (w // g), g * g)

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        f = self.features(pixel_values, self.patch_size, self.pool)
        c = self.centroids.to(f.device)
        d = (f.pow(2).sum(-1, keepdim=True) - 2 * f @ c.T + c.pow(2).sum(-1))  # B,N,K
        return d.argmin(-1)

    @classmethod
    def fit(cls, pixel_batches, vocab_size: int = 512, patch_size: int = 16, pool: int = 4,
            seed: int = 0) -> "KMeansPatchTokenizer":
        from sklearn.cluster import MiniBatchKMeans

        feats = np.concatenate([cls.features(pv, patch_size, pool).reshape(-1, pool * pool).numpy()
                                for pv in pixel_batches])
        km = MiniBatchKMeans(n_clusters=vocab_size, random_state=seed, batch_size=4096, n_init=3)
        km.fit(feats)
        return cls(km.cluster_centers_, patch_size, pool)

    def save(self, path: str | Path) -> None:
        np.savez(path, centroids=self.centroids.numpy(), patch_size=self.patch_size, pool=self.pool)

    @classmethod
    def load(cls, path: str | Path) -> "KMeansPatchTokenizer":
        z = np.load(path)
        return cls(z["centroids"], int(z["patch_size"]), int(z["pool"]))
