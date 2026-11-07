"""Turning (image, words, boxes) documents into masked pre-training batches."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import get_worker_info

from .masking import blockwise_mask, span_mask, wpa_labels


def encode_documents(processor, examples, max_length: int = 512):
    """Tokenise a batch of documents with ``LayoutLMv3Processor`` (no OCR: words and 0-1000 boxes given)."""
    images = [im.convert("RGB") for im in examples["image"]]
    return processor(images, examples["tokens"], boxes=examples["bboxes"], truncation=True,
                     padding="max_length", max_length=max_length, return_tensors="pt")


def add_mim_labels(batch: dict[str, torch.Tensor], image_tokenizer) -> dict[str, torch.Tensor]:
    """MIM targets: visual tokens of the clean image at the masked patches, -100 elsewhere."""
    visual_tokens = image_tokenizer.encode(batch["pixel_values"]).to(batch["bool_masked_pos"].device)
    batch["mim_labels"] = torch.where(batch["bool_masked_pos"], visual_tokens,
                                      torch.full_like(visual_tokens, -100))
    return batch


class PretrainingCollator:
    """Applies MLM span masking, MIM blockwise masking and builds WPA labels for each example.

    If ``image_tokenizer`` is None, ``mim_labels`` are left out so that they can be
    computed later on the GPU with ``add_mim_labels`` (useful for a dVAE tokenizer
    and with DataLoader worker processes).

    Random masks: every DataLoader worker gets a copy of the collator, so the random generators are created lazily,
    in the process that uses them. In the main process they are seeded with ``seed`` (an int seed gives the same
    masks as before this was lazy; validation batches are collated there), in worker ``w`` with ``(seed, w)``, so
    workers draw different masks and a run is still reproducible. ``seed`` may be an int, a sequence of ints
    (e.g. ``(seed, resume_step)``) or None (fresh entropy).
    """

    def __init__(self, tokenizer, image_tokenizer=None, mlm_ratio: float = 0.3, mim_ratio: float = 0.4,
                 poisson_lambda: float = 3.0, grid: int = 14, seed: int | Sequence[int] | None = None):
        self.tok, self.img_tok = tokenizer, image_tokenizer
        self.mlm_ratio, self.mim_ratio, self.lam, self.grid = mlm_ratio, mim_ratio, poisson_lambda, grid
        self.seed = seed
        self._rng_owner = None

    def _seed_rngs(self) -> None:
        info = get_worker_info()
        owner = info.id if info is not None else None
        if self._rng_owner == ("set", owner):
            return
        self._rng_owner = ("set", owner)
        if self.seed is None:
            self.np_rng, self.py_rng = np.random.default_rng(), random.Random()
            return
        entropy = list(self.seed) if isinstance(self.seed, Sequence) else [self.seed]
        if owner is None and len(entropy) == 1:
            self.np_rng, self.py_rng = np.random.default_rng(entropy[0]), random.Random(entropy[0])
            return
        if owner is not None:
            entropy.append(1_000_003 + owner)                  # worker w: its own stream
        self.np_rng = np.random.default_rng(entropy)
        self.py_rng = random.Random(int(np.random.SeedSequence(entropy + [7]).generate_state(1, np.uint64)[0]))

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        self._seed_rngs()
        batch = {k: torch.stack([torch.as_tensor(f[k]) for f in features])
                 for k in ("input_ids", "bbox", "attention_mask", "pixel_values")}
        input_ids = batch["input_ids"].clone()
        mlm_labels = torch.full_like(input_ids, -100)
        wpa = torch.full_like(input_ids, -100)
        image_masks = []
        special_ids = set(self.tok.all_special_ids)
        for b in range(input_ids.shape[0]):
            ids = batch["input_ids"][b]
            special = torch.tensor([int(i) in special_ids for i in ids.tolist()])
            cand = torch.nonzero(~special).flatten().tolist()
            masked = span_mask(cand, self.mlm_ratio, self.lam, self.np_rng)
            text_mask = torch.zeros(len(ids), dtype=torch.bool)
            text_mask[masked] = True
            mlm_labels[b, text_mask] = ids[text_mask]
            input_ids[b, text_mask] = self.tok.mask_token_id
            img_mask = torch.from_numpy(blockwise_mask(self.grid, self.grid, self.mim_ratio, rng=self.py_rng))
            image_masks.append(img_mask.reshape(-1))
            wpa[b] = wpa_labels(batch["bbox"][b], text_mask, img_mask, special, self.grid)
        batch.update(input_ids=input_ids, mlm_labels=mlm_labels, wpa_labels=wpa,
                     bool_masked_pos=torch.stack(image_masks))
        if self.img_tok is not None:
            add_mim_labels(batch, self.img_tok)                                # targets from the clean image
        return batch
