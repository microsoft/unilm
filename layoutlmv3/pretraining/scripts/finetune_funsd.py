"""Fine-tune a LayoutLMv3 checkpoint on entity labelling and report entity-level F1 (seqeval).

Tasks:
  * funsd: FUNSD (nielsr/funsd-layoutlmv3), segment-level boxes as provided
  * cord:  CORD v2 receipts (naver-clova-ix/cord-v2, CC BY 4.0), converted here to words,
           word-level 0-1000 boxes and BIO tags over its 29 line categories

Example:
    python scripts/finetune_funsd.py --model microsoft/layoutlmv3-base --seeds 1 2 3 4 5
    python scripts/finetune_funsd.py --task cord --model runs/funsd_pt/backbone --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import ClassLabel, Features, Sequence, Value, load_dataset
from datasets import Image as ImageFeature
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from layoutlmv3_pretraining.tracking import RunTracker, write_json_atomic
from transformers import (LayoutLMv3ForTokenClassification, LayoutLMv3Processor,
                          get_linear_schedule_with_warmup)

DEFAULTS = {"funsd": dict(steps=1000, lr=1e-5, batch_size=2), "cord": dict(steps=1000, lr=5e-5, batch_size=4)}
# --protocol strict / paper: see --help. paper: the LayoutLMv3 paper's settings (Sec. 3.3: 1,000 steps; FUNSD lr 1e-5, batch 16; CORD lr 5e-5,
# batch 64), labels on the first sub-token of each word only, and words cut off by the 512-token limit scored as
# missed (predicted "O") instead of being dropped from the evaluation.
PAPER = {"funsd": dict(steps=1000, lr=1e-5, batch_size=16), "cord": dict(steps=1000, lr=5e-5, batch_size=64)}


def encode(processor, examples, first_subword_only=False):
    images = [im.convert("RGB") for im in examples["image"]]
    enc = processor(images, examples["tokens"], boxes=examples["bboxes"], word_labels=examples["ner_tags"],
                    truncation=True, padding="max_length", max_length=512)
    if first_subword_only:  # the tokenizer's own rule misses some words (e.g. "\u2610"/"\u2611" split in 2 tokens)
        for i, labels in enumerate(enc["labels"]):
            prev = None
            for j, w in enumerate(enc.word_ids(i)):
                if w is None or w == prev:
                    labels[j] = -100
                elif labels[j] == -100:
                    raise ValueError(f"first sub-token of word {w} has no label")
                if w is not None:
                    prev = w
    return enc


def truncated_tags(processor, split, names):
    """Gold tags of the words that the 512-token limit cuts off, per example (scored as missed in --protocol paper)."""
    tails = []
    for ex in split:
        enc = processor.tokenizer(ex["tokens"], boxes=ex["bboxes"], truncation=True, max_length=512)
        kept = max(w for w in enc.word_ids() if w is not None) + 1
        tails.append([names[t] for t in ex["ner_tags"][kept:]])
    return tails


def cord_to_token_classification(ground_truth, label2id):
    """One CORD v2 example -> words, word-level 0-1000 boxes and BIO tags (first word of a line: B-)."""
    gt = json.loads(ground_truth)
    w, h = gt["meta"]["image_size"]["width"], gt["meta"]["image_size"]["height"]
    tokens, boxes, tags = [], [], []
    for line in gt["valid_line"]:
        cat = line["category"]
        words = [x for x in line["words"] if x["text"].strip()]
        for j, word in enumerate(words):
            q = word["quad"]
            x0, x1 = min(q["x1"], q["x4"]), max(q["x2"], q["x3"])
            y0, y1 = min(q["y1"], q["y2"]), max(q["y3"], q["y4"])
            box = [int(1000 * min(max(x0 / w, 0), 1)), int(1000 * min(max(y0 / h, 0), 1)),
                   int(1000 * min(max(x1 / w, 0), 1)), int(1000 * min(max(y1 / h, 0), 1))]
            box[2], box[3] = max(box[2], box[0]), max(box[3], box[1])
            tokens.append(word["text"]); boxes.append(box)
            tags.append(label2id[("B-" if j == 0 else "I-") + cat.upper()])
    return {"tokens": tokens, "bboxes": boxes, "ner_tags": tags}


def load_task(task):
    if task == "funsd":
        return load_dataset("nielsr/funsd-layoutlmv3")
    raw = load_dataset("naver-clova-ix/cord-v2")
    cats = sorted({l["category"].upper() for s in raw for gt in raw[s]["ground_truth"]
                   for l in json.loads(gt)["valid_line"]})
    names = ["O"] + [f"{p}-{c}" for c in cats for p in ("B", "I")]
    label2id = {n: i for i, n in enumerate(names)}
    feats = Features({"tokens": Sequence(Value("string")), "bboxes": Sequence(Sequence(Value("int64"))),
                      "ner_tags": Sequence(ClassLabel(names=names)), "image": ImageFeature()})
    # only the JSON column is read, so the images are copied without being decoded and re-encoded
    return raw.map(lambda g: cord_to_token_classification(g, label2id), input_columns="ground_truth",
                   remove_columns=["ground_truth"], features=feats)


def evaluate(model, test, labels, device, tails=None):
    """Entity F1 (seqeval) and mean token loss on the test split."""
    was_training = model.training
    model.eval(); y_true, y_pred, losses = [], [], []
    with torch.no_grad():
        for b in DataLoader(test, batch_size=8):
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**b)
            losses.append(out.loss.item())
            pred = out.logits.argmax(-1)
            for p, l in zip(pred.cpu().tolist(), b["labels"].cpu().tolist()):
                keep = [(labels[pi], labels[li]) for pi, li in zip(p, l) if li != -100]
                y_pred.append([x for x, _ in keep]); y_true.append([y for _, y in keep])
    if tails is not None:  # words beyond the 512-token limit: gold tags kept, predicted "O"
        assert len(tails) == len(y_true)
        for t, p, tail in zip(y_true, y_pred, tails):
            t.extend(tail); p.extend(["O"] * len(tail))
    if was_training:
        model.train()
    return f1_score(y_true, y_pred), float(np.mean(losses))


def run(ds, model_path, seed, steps, lr, batch_size, device, eval_at=(), paper=False, micro_batch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    labels = ds["train"].features["ner_tags"].feature.names
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    cols = ds["train"].column_names
    train = ds["train"].map(lambda e: encode(processor, e, paper), batched=True, remove_columns=cols)
    test = ds["test"].map(lambda e: encode(processor, e, paper), batched=True, remove_columns=cols)
    tails = truncated_tags(processor, ds["test"], labels) if paper else None
    train.set_format("torch"); test.set_format("torch")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_path, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)}).to(device)
    g = torch.Generator().manual_seed(seed)
    accum = max(1, batch_size // micro_batch) if paper else 1
    assert batch_size % accum == 0
    loader = DataLoader(train, batch_size=batch_size // accum, shuffle=True, generator=g, drop_last=paper)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    # Learning curve (--eval_at): evaluation runs under no_grad in eval mode and uses no random numbers, so the
    # training trajectory and the final F1 are the same as without it.
    curve, window = [], []
    if 0 in eval_at:
        f1, loss = evaluate(model, test, labels, device, tails)
        curve.append({"step": 0, "train_loss": None, "test_loss": round(loss, 4), "test_f1": round(f1, 4)})
    model.train(); step = 0; micro = 0
    while step < steps:
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            (loss / accum).backward(); window.append(loss.item()); micro += 1
            if micro % accum:
                continue
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step in eval_at and step < steps:
                f1, tl = evaluate(model, test, labels, device, tails)
                curve.append({"step": step, "train_loss": round(float(np.mean(window)), 4),
                              "test_loss": round(tl, 4), "test_f1": round(f1, 4)})
                window = []
            if step >= steps:
                break
    f1, tl = evaluate(model, test, labels, device, tails)
    if eval_at:
        curve.append({"step": steps, "train_loss": round(float(np.mean(window)), 4) if window else None,
                      "test_loss": round(tl, 4), "test_f1": round(f1, 4)})
    return f1, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["funsd", "cord"], default="funsd")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--out")
    ap.add_argument("--protocol", choices=["legacy", "strict", "paper"], default="legacy",
                    help="legacy: the settings of all results so far; strict: legacy batch sizes with first-sub-token "
                         "labels and truncated words scored as missed; paper: strict plus the paper's batch sizes (PAPER)")
    ap.add_argument("--micro_batch", type=int, default=16, help="--protocol paper: per-forward batch (accumulation)")
    ap.add_argument("--eval_at", type=int, nargs="*", default=[],
                    help="also evaluate at these fine-tuning steps (0 = before training) and save the learning curve")
    ap.add_argument("--tags", nargs="*", default=[], help="free-form tags for experiments/registry.jsonl")
    a = ap.parse_args()
    base_hp = PAPER if a.protocol == "paper" else DEFAULTS
    hp = {k: getattr(a, k) if getattr(a, k) is not None else v for k, v in base_hp[a.task].items()}
    out_dir = os.path.dirname(os.path.abspath(a.out)) if a.out else None
    with RunTracker(f"finetune_{a.task}", a, tags=a.tags) as tracker:
        tracker.config.update(task=a.task, model=a.model, seeds=a.seeds, protocol=a.protocol, micro_batch=a.micro_batch, **hp,
                              dataset="nielsr/funsd-layoutlmv3" if a.task == "funsd" else "naver-clova-ix/cord-v2")
        tracker.outputs["results"] = os.path.abspath(a.out) if a.out else None
        res = finetune(a, hp, tracker)
        tracker.metrics.update({k: res[k] for k in ("f1", "mean", "std", "minutes")})
    if a.out:  # the full record next to the results file
        stem = os.path.splitext(os.path.basename(a.out))[0]
        write_json_atomic(os.path.join(out_dir, "records", f"{stem}.run_record.json"), tracker.record)


def finetune(a, hp, tracker):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_task(a.task)
    t0 = time.time()
    f1s, curves = [], {}
    for s in a.seeds:
        f1, curve = run(ds, a.model, s, device=device, eval_at=set(a.eval_at), paper=a.protocol != "legacy",
                        micro_batch=a.micro_batch, **hp)
        f1s.append(f1)
        if curve:
            curves[s] = curve
            print(json.dumps({"seed": s, "curve": curve}), flush=True)
            for point in curve:  # W&B steps must increase, so the fine-tuning step is logged as a metric
                tracker.log({"seed": s, "finetune_step": point["step"],
                             **{f"curve/{k}": v for k, v in point.items() if k != "step" and v is not None}})
        tracker.log({"seed": s, f"{a.task}/f1": f1})
        tracker.metrics.setdefault("per_seed", {})[s] = round(f1s[-1], 4)
        print(json.dumps({"seed": s, "f1": round(f1s[-1], 4)}), flush=True)
    res = {"task": a.task, "model": a.model, "protocol": a.protocol, **hp, "seeds": a.seeds, "f1": [round(f, 4) for f in f1s],
           "mean": round(float(np.mean(f1s)), 4),
           "std": round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else 0.0,
           "minutes": round((time.time() - t0) / 60, 1)}
    if curves:
        res["curves"] = curves
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
