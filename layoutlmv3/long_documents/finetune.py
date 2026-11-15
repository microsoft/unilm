"""Fine-tune LayoutLMv3 on whole document pages (token classification) and report entity F1 (seqeval).

The data is a ``datasets.DatasetDict`` saved with ``save_to_disk`` (``train`` and ``test`` splits; columns ``tokens``,
``bboxes`` with 0-1000 word boxes, ``ner_tags`` as BIO ``ClassLabel``s, and ``image``), as written by
``prepare_vrdu.py`` and ``prepare_realkie.py``.

Scoring is strict: every word of every test page is scored, and words beyond ``--max_length`` tokens, which the
model never sees, count as predicted "O". Labels sit on the first sub-token of each word.

Examples:
    # baseline: pages cut at 512 tokens
    python finetune.py --data data/vrdu_adbuy --max_length 512
    # whole pages up to 2048 tokens
    python finetune.py --data data/vrdu_adbuy --max_length 2048 --gradient_checkpointing
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from datasets import load_from_disk
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, get_linear_schedule_with_warmup

from long_documents import extend_position_embeddings, use_lut_attention


def encode(processor, examples, max_length):
    images = [im.convert("RGB") for im in examples["image"]]
    enc = processor(images, examples["tokens"], boxes=examples["bboxes"], word_labels=examples["ner_tags"],
                    truncation=True, max_length=max_length)
    for i, labels in enumerate(enc["labels"]):   # label only the first sub-token of each word
        prev = None
        for j, w in enumerate(enc.word_ids(i)):
            if w is None or w == prev:
                labels[j] = -100
            if w is not None:
                prev = w
    return enc


def cut_off_tags(processor, split, names, max_length):
    """Gold tags of the words beyond ``max_length`` tokens, per page."""
    tails = []
    for ex in split:
        enc = processor.tokenizer(ex["tokens"], boxes=ex["bboxes"], truncation=True, max_length=max_length)
        kept = max(w for w in enc.word_ids() if w is not None) + 1
        tails.append([names[t] for t in ex["ner_tags"][kept:]])
    return tails


def collate(pad_id):
    """Pad each batch to its longest page only."""
    def fn(features):
        n = max(len(f["input_ids"]) for f in features)
        out = {}
        for key, value in (("input_ids", pad_id), ("attention_mask", 0), ("bbox", [0, 0, 0, 0]), ("labels", -100)):
            out[key] = torch.tensor([list(f[key]) + [value] * (n - len(f[key])) for f in features])
        out["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        return out
    return fn


@torch.no_grad()
def evaluate(model, test, tails, names, device, fn):
    model.eval()
    y_true, y_pred = [], []
    for b in DataLoader(test, batch_size=4, collate_fn=fn):
        b = {k: v.to(device) for k, v in b.items()}
        pred = model(**b).logits.argmax(-1)
        for p, gold in zip(pred.tolist(), b["labels"].tolist()):
            keep = [(names[pi], names[gi]) for pi, gi in zip(p, gold) if gi != -100]
            y_pred.append([x for x, _ in keep])
            y_true.append([y for _, y in keep])
    for t, p, tail in zip(y_true, y_pred, tails):
        t.extend(tail)
        p.extend(["O"] * len(tail))
    model.train()
    return f1_score(y_true, y_pred)


def run(ds, a, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    names = ds["train"].features["ner_tags"].feature.names
    processor = LayoutLMv3Processor.from_pretrained(a.model, apply_ocr=False)
    cols = ds["train"].column_names
    train = ds["train"].map(lambda e: encode(processor, e, a.max_length), batched=True, remove_columns=cols)
    test = ds["test"].map(lambda e: encode(processor, e, a.max_length), batched=True, remove_columns=cols)
    tails = cut_off_tags(processor, ds["test"], names, a.max_length)
    for split in (train, test):   # images as tensors; the text columns are padded per batch by `collate`
        split.set_format("torch", columns=["pixel_values"], output_all_columns=True)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        a.model, num_labels=len(names), id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)})
    if a.max_length > 512:
        extend_position_embeddings(model, a.max_length)
    use_lut_attention(model)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device).train()
    fn = collate(processor.tokenizer.pad_token_id)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, generator=g, collate_fn=fn)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * a.steps), a.steps)
    step = 0
    while step < a.steps:
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            model(**b).loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step >= a.steps:
                break
    return evaluate(model, test, tails, names, device, fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="DatasetDict directory from prepare_vrdu.py / prepare_realkie.py")
    ap.add_argument("--model", default="microsoft/layoutlmv3-base")
    ap.add_argument("--max_length", type=int, default=2048, help="text tokens per page; 512 = the usual truncation")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--out", help="write the per-seed F1 to this JSON file")
    a = ap.parse_args()
    ds = load_from_disk(a.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0, f1s = time.time(), []
    for s in a.seeds:
        f1s.append(round(run(ds, a, s, device), 4))
        print(f"seed {s}: F1 {f1s[-1]}", flush=True)
    result = {"data": a.data, "model": a.model, "max_length": a.max_length, "seeds": a.seeds, "f1": f1s,
              "mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s)), 4),
              "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(result))
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
