"""Masking strategies from the LayoutLMv3 paper (Huang et al., 2022, Section 2.2).

* Text: span masking of 30% of text tokens, span lengths ~ Poisson(lambda=3).
* Image: blockwise masking of 40% of image patches (as in BEiT).
* Word-patch alignment labels derived from both masks.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch


def span_mask(candidates: list[int], ratio: float = 0.3, poisson_lambda: float = 3.0,
              rng: np.random.Generator | None = None) -> list[int]:
    """Choose ~ratio of `candidates` (token positions) as contiguous spans.

    Spans are contiguous in `candidates` order; lengths are drawn from
    Poisson(poisson_lambda), with zero-length draws resampled to 1.
    """
    rng = rng or np.random.default_rng()
    n = len(candidates)
    target = int(round(n * ratio))
    if target == 0:
        return []
    masked: set[int] = set()
    attempts = 0
    while len(masked) < target and attempts < 10 * n:
        attempts += 1
        length = max(1, int(rng.poisson(poisson_lambda)))
        length = min(length, target - len(masked))
        start = int(rng.integers(0, n))
        for k in range(start, min(n, start + length)):
            masked.add(candidates[k])
    return sorted(masked)


def blockwise_mask(height: int = 14, width: int = 14, ratio: float = 0.4, min_block: int = 16,
                   min_aspect: float = 0.3, rng: random.Random | None = None) -> np.ndarray:
    """BEiT-style blockwise masking: returns a (height, width) bool array with ~ratio True."""
    rng = rng or random.Random()
    num_masking = int(round(height * width * ratio))
    mask = np.zeros((height, width), dtype=bool)
    log_aspect = (math.log(min_aspect), math.log(1 / min_aspect))
    count = 0
    while count < num_masking:
        max_mask = num_masking - count
        delta = 0
        for _ in range(10):
            target_area = rng.uniform(min_block, max(min_block, max_mask))
            aspect = math.exp(rng.uniform(*log_aspect))
            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))
            if 0 < w < width and 0 < h < height:
                top, left = rng.randint(0, height - h), rng.randint(0, width - w)
                block = mask[top:top + h, left:left + w]
                new = h * w - int(block.sum())
                if 0 < new <= max_mask:
                    block[:] = True
                    delta = new
                if delta > 0:
                    break
        if delta == 0:
            # no block fits the remaining budget: mask single random patches
            free = np.argwhere(~mask)
            for r, c in free[rng.sample(range(len(free)), min(max_mask, len(free)))]:
                mask[r, c] = True
            break
        count += delta
    return mask


def bbox_to_patches(box: list[int] | torch.Tensor, grid: int = 14, scale: int = 1000) -> list[int]:
    """Indices (row-major) of the patches in a grid x grid layout that a 0-1000 box overlaps."""
    x0, y0, x1, y1 = [int(v) for v in box]
    if x1 <= x0 or y1 <= y0:
        return []
    c0, c1 = x0 * grid // scale, min(grid - 1, (x1 * grid - 1) // scale)
    r0, r1 = y0 * grid // scale, min(grid - 1, (y1 * grid - 1) // scale)
    return [r * grid + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def wpa_labels(bbox: torch.Tensor, text_mask: torch.Tensor, image_mask: torch.Tensor,
               special: torch.Tensor, grid: int = 14) -> torch.Tensor:
    """Word-patch alignment labels for one sequence.

    1 ("aligned")   : unmasked text token whose covered image patches are all unmasked
    0 ("unaligned") : unmasked text token with at least one covered patch masked
    -100 (ignored)  : masked text tokens, special/padding tokens, tokens without a box
    """
    labels = torch.full(text_mask.shape, -100, dtype=torch.long)
    flat = image_mask.reshape(-1)
    for i in range(len(labels)):
        if special[i] or text_mask[i]:
            continue
        patches = bbox_to_patches(bbox[i].tolist(), grid)
        if not patches:
            continue
        labels[i] = 0 if bool(flat[patches].any()) else 1
    return labels
