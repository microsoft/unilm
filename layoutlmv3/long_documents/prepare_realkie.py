"""Convert a RealKIE dataset (IndicoDataSolutions/RealKIE, e.g. FCC Invoices) into page-level token classification.

RealKIE ships, per document, the OCR (one entry per page, each token with its pixel box and character offsets into the
document text), the page images and the labels as character spans into that text. Each page becomes one example in
the same format as prepare_vrdu.py (words, 0-1000 boxes, BIO tags, page image), so finetune.py --data can train on
the output.

The data is CC BY-NC 4.0.

Example:
    python prepare_realkie.py --realkie_dir realkie --dataset fcc_invoices --out data/realkie_fcc_invoices
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from datasets import Image as ImageFeature
from PIL import Image

csv.field_size_limit(sys.maxsize)


def field_tag(name: str) -> str:
    """'Line Item - Start Date' -> 'LINE_ITEM_START_DATE'."""
    return "_".join(name.upper().replace("-", " ").split())


def page_example(page, spans, image, labels):
    """`spans`: [(tag, start, end)] in document character offsets. A word gets the first span it overlaps."""
    width, height = image.size
    words, boxes, tags = [], [], []
    used = set()
    for tok in page["tokens"]:
        text = tok["text"].strip()
        if not text:
            continue
        pos, off = tok["position"], tok["doc_offset"]
        box = [int(1000 * min(max(v / size, 0.0), 1.0))
               for v, size in ((pos["left"], width), (pos["top"], height), (pos["right"], width), (pos["bottom"], height))]
        box[2], box[3] = max(box[2], box[0]), max(box[3], box[1])
        tag = "O"
        for i, (name, s, e) in enumerate(spans):
            if off["start"] < e and off["end"] > s:
                tag = ("I-" if i in used else "B-") + name
                used.add(i)
                break
        words.append(text); boxes.append(box); tags.append(labels.index(tag))
    return {"tokens": words, "bboxes": boxes, "ner_tags": tags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realkie_dir", required=True, help="directory that contains the unzipped dataset folder")
    ap.add_argument("--dataset", default="fcc_invoices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_side", type=int, default=1000, help="longest side of the saved page images, in pixels")
    a = ap.parse_args()

    splits = {"train": "train.csv", "valid": "val.csv", "test": "test.csv"}
    rows = {name: list(csv.DictReader(open(os.path.join(a.realkie_dir, a.dataset, f)))) for name, f in splits.items()}
    fields = sorted({field_tag(l["label"]) for r in rows["train"] for l in json.loads(r["labels"])})
    labels = ["O"] + [f"{p}-{f}" for f in fields for p in ("B", "I")]
    feats = Features({"tokens": Sequence(Value("string")), "bboxes": Sequence(Sequence(Value("int64"))),
                      "ner_tags": Sequence(ClassLabel(names=labels)), "image": ImageFeature(),
                      "doc": Value("string"), "page": Value("int64")})
    out, stats = {}, {}
    for name, split_rows in rows.items():
        examples = []
        for r in split_rows:
            spans = [(field_tag(l["label"]), l["start"], l["end"]) for l in json.loads(r["labels"])
                     if field_tag(l["label"]) in fields]
            ocr = json.load(gzip.open(os.path.join(a.realkie_dir, r["ocr"])))
            images = json.loads(r["image_files"])
            for i, page in enumerate(ocr):
                image = Image.open(os.path.join(a.realkie_dir, images[i])).convert("RGB")
                ex = page_example(page, spans, image, labels)
                if not ex["tokens"]:
                    continue
                image.thumbnail((a.max_side, a.max_side))
                examples.append({**ex, "image": image, "doc": r["original_filename"], "page": i})
        out[name] = Dataset.from_list(examples, features=feats)
        tagged = np.mean([any(t != 0 for t in e["ner_tags"]) for e in examples])
        stats[name] = {"documents": len(split_rows), "pages": len(examples), "pages_with_fields": round(float(tagged), 3)}
        print(name, stats[name], flush=True)
    DatasetDict(out).save_to_disk(a.out)
    json.dump({"labels": labels, "stats": stats, "source": f"RealKIE/{a.dataset}", "license": "CC BY-NC 4.0"},
              open(os.path.join(a.out, "prepare_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
