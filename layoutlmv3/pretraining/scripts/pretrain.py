"""Continued pre-training of LayoutLMv3 with MLM + MIM + WPA.

Two corpora:
  * ``--corpus hf``  a small Hugging Face dataset with words and 0-1000 boxes (default: FUNSD train)
  * ``--corpus idl`` pages from the Industry Documents Library (pixparse/idl-wds), streamed shard by shard

Two MIM target tokenizers:
  * ``--image_tokenizer kmeans`` a k-means patch codebook fitted on training images (no external weights)
  * ``--image_tokenizer dalle``  OpenAI's DALL-E dVAE encoder (8192 tokens, on a 112x112 view, as in BEiT)

Examples:
    python scripts/pretrain.py --output_dir runs/funsd_pt --max_steps 3000
    python scripts/pretrain.py --corpus idl --idl_shards 0-26 --image_tokenizer dalle \\
        --output_dir runs/idl_dalle --max_steps 6000 --batch_size 16 --grad_accum 2 --num_workers 12

``output_dir/backbone`` holds the checkpoint with the lowest validation loss (a plain
``LayoutLMv3Model``); ``output_dir/last_backbone`` the final one if ``--save_last``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor, get_cosine_schedule_with_warmup

from layoutlmv3_pretraining import DalleTokenizer, KMeansPatchTokenizer, LayoutLMv3ForPreTraining
from layoutlmv3_pretraining.corpus import IDLPages, NgramBlocklist, resolve_idl_sources
from layoutlmv3_pretraining.data import PretrainingCollator, add_mim_labels, encode_documents
from layoutlmv3_pretraining.modeling import DEFAULT_WPA_HEAD
from layoutlmv3_pretraining.tracking import RunTracker


ADAM_BETAS = (0.9, 0.98)  # as in the LayoutLMv3 paper (the PyTorch default before scale-v2 was (0.9, 0.999))


def parse_range(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def accuracy(logits, labels):
    m = labels != -100
    return (logits.argmax(-1)[m] == labels[m]).float().mean().item() if m.any() else float("nan")


def to_device(batch, device, img_tok):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    if "mim_labels" not in batch:
        add_mim_labels(batch, img_tok)
    return batch


@torch.no_grad()
def evaluate(model, val_batches, device, img_tok):
    """Mean of per-batch losses/accuracies, plus pooled WPA AUC / balanced accuracy and MIM target statistics."""
    model.eval()
    agg = {k: [] for k in ("loss", "mlm_loss", "mim_loss", "wpa_loss", "mlm_acc", "mim_acc", "wpa_acc")}
    wpa_p, wpa_y, mim_y = [], [], []
    for batch in val_batches:
        batch = to_device(batch, device, img_tok)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(**batch)
        for k in ("loss", "mlm_loss", "mim_loss", "wpa_loss"):
            agg[k].append(getattr(out, k).item())
        agg["mlm_acc"].append(accuracy(out.mlm_logits, batch["mlm_labels"]))
        agg["mim_acc"].append(accuracy(out.mim_logits, batch["mim_labels"]))
        agg["wpa_acc"].append(accuracy(out.wpa_logits, batch["wpa_labels"]))
        m = batch["wpa_labels"] != -100
        wpa_p.append(out.wpa_logits.float().softmax(-1)[..., 1][m].cpu()); wpa_y.append(batch["wpa_labels"][m].cpu())
        mim_y.append(batch["mim_labels"][batch["mim_labels"] != -100].cpu())
    model.train()
    res = {k: round(sum(v) / len(v), 4) for k, v in agg.items()}
    p, y = torch.cat(wpa_p).numpy(), torch.cat(wpa_y).numpy()
    res["wpa_auc"] = round(float(roc_auc_score(y, p)), 4) if len(set(y.tolist())) == 2 else float("nan")
    res["wpa_bal_acc"] = round(float(balanced_accuracy_score(y, p > 0.5)), 4)
    res["wpa_majority"] = round(float(max(y.mean(), 1 - y.mean())), 4)
    counts = torch.bincount(torch.cat(mim_y)).float()
    probs = counts[counts > 0] / counts.sum()
    res["mim_codes_used"] = int((counts > 0).sum())
    res["mim_code_perplexity"] = round(float(torch.exp(-(probs * probs.log()).sum())), 1)
    res["mim_majority"] = round(float(probs.max()), 4)
    return res


def save_backbone(model, processor, path: Path):
    model.layoutlmv3.save_pretrained(path)
    processor.save_pretrained(path)


def save_train_state(path: Path, **state):
    """Everything needed to continue training (``--resume``); written atomically so a kill never leaves half a file."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def build_hf_corpus(a, processor):
    raw = load_dataset(a.dataset, split=a.split).shuffle(seed=a.seed)
    enc = raw.map(lambda ex: encode_documents(processor, ex), batched=True, batch_size=16,
                  remove_columns=raw.column_names)
    enc.set_format("torch")
    val, train = enc.select(range(a.val_docs)), enc.select(range(a.val_docs, len(enc)))
    fit_images = (train[i:i + 16]["pixel_values"] for i in range(0, len(train), 16))
    return train, list(val), fit_images, {"train_docs": len(train), "val_docs": len(val)}


def build_idl_corpus(a, processor):
    exclude = None
    if a.exclude_funsd_test:
        funsd = load_dataset("nielsr/funsd-layoutlmv3", split="test")
        exclude = NgramBlocklist(funsd["tokens"], n=5, threshold=0.5)
    common = dict(max_pages_per_doc=a.pages_per_doc, min_words=a.min_words, exclude=exclude,
                  orientation=a.idl_orientation)
    train = IDLPages(resolve_idl_sources(parse_range(a.idl_shards), download=not a.idl_stream), processor,
                     max_pages=a.max_pages, shuffle_buffer=a.shuffle_buffer, repeat=True, seed=a.data_seed, **common)
    val_ds = IDLPages(resolve_idl_sources(parse_range(a.idl_val_shards), download=not a.idl_stream), processor,
                      max_pages=a.val_docs, **common)
    val = list(val_ds)
    if a.resume_batches:  # continue the page stream where the interrupted run stopped (same seed, same workers)
        train.skip_batches(a.resume_batches, a.batch_size)

    def fit_images():
        it = iter(IDLPages(train.sources, processor, max_pages=a.kmeans_fit_pages, **common))
        while chunk := [x["pixel_values"] for _, x in zip(range(16), it)]:
            yield torch.stack(chunk)

    info = {"train_shards": a.idl_shards, "val_shards": a.idl_val_shards, "val_pages": len(val),
            "orientation": a.idl_orientation, "val_pages_turned": sum(1 for x in val if x["rotation"]),
            "val_keys": [x["key"] for x in val]}
    if a.count_pages:  # one cheap pass over the OCR json only (no image decoding, so no orientation check)
        info["train_pages"] = sum(1 for _ in IDLPages(train.sources, processor, max_pages=a.max_pages,
                                                      **common).raw_pages(decode_images=False))
        if exclude is not None:
            info["funsd_test_overlap_excluded"] = sum(
                1 for _ in IDLPages(train.sources, processor, max_pages_per_doc=a.pages_per_doc,
                                    min_words=a.min_words, exclude=lambda w: not exclude(w)
                                    ).raw_pages(decode_images=False))
    return train, val, fit_images(), info


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/layoutlmv3-base")
    ap.add_argument("--output_dir", required=True)
    # corpus
    ap.add_argument("--corpus", choices=["hf", "idl"], default="hf")
    ap.add_argument("--dataset", default="nielsr/funsd-layoutlmv3", help="--corpus hf: dataset with image/tokens/bboxes")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val_docs", type=int, default=20, help="documents/pages held out to monitor the objectives")
    ap.add_argument("--idl_shards", default="0-26", help="--corpus idl: training shards, e.g. 0-26 or 0,3,5-9")
    ap.add_argument("--idl_val_shards", default="2999", help="--corpus idl: shards for held-out pages")
    ap.add_argument("--idl_stream", action="store_true", help="stream shards over HTTP instead of caching them")
    ap.add_argument("--pages_per_doc", type=int, default=2)
    ap.add_argument("--min_words", type=int, default=10)
    ap.add_argument("--max_pages", type=int, default=None, help="cap on training pages (default: all in the shards)")
    ap.add_argument("--shuffle_buffer", type=int, default=500)
    ap.add_argument("--exclude_funsd_test", action="store_true",
                    help="drop IDL pages whose 5-grams largely occur in FUNSD test documents")
    ap.add_argument("--count_pages", action="store_true",
                    help="count training pages before training (logged; from the OCR json only, so pages dropped by "
                         "the orientation check are still counted)")
    ap.add_argument("--idl_orientation", choices=["fix", "drop", "off"], default="fix",
                    help="IDL frames stored turned relative to their OCR boxes: fix (turn 90/270-degree pages upright, "
                         "drop 180-degree and ambiguous ones), drop (drop all of them) or off (as stored; the "
                         "behaviour before scale-v2)"),
    # MIM targets
    ap.add_argument("--image_tokenizer", choices=["kmeans", "dalle"], default="kmeans")
    ap.add_argument("--image_vocab_size", type=int, default=512, help="k-means codebook size (dalle: 8192)")
    ap.add_argument("--kmeans_fit_pages", type=int, default=2000)
    ap.add_argument("--dalle_encoder", default=None,
                    help="path to DALL-E encoder.pkl (or a state_dict .pt); downloaded from OpenAI if omitted")
    # optimisation
    ap.add_argument("--max_steps", type=int, default=2000, help="optimiser steps")
    ap.add_argument("--batch_size", type=int, default=8, help="per forward pass")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5, help="peak learning rate of the top encoder layer")
    ap.add_argument("--layer_decay", type=float, default=1.0,
                    help="layer-wise lr decay: layer i gets lr * decay ** (L - i), embeddings lr * decay ** (L + 1)")
    ap.add_argument("--new_params_lr", type=float, default=None,
                    help="lr of the heads and image [MASK] token after warm-up (default: --lr)")
    ap.add_argument("--head_warmup_steps", type=int, default=0,
                    help="optimiser steps that train only the new parameters (heads, image [MASK] token) "
                         "with the encoder frozen, before joint training; not counted in --max_steps")
    ap.add_argument("--head_lr", type=float, default=1e-3, help="peak lr during the head warm-up (cosine)")
    ap.add_argument("--wpa_head", choices=["mlp", "linear", "mlp_patch"], default=DEFAULT_WPA_HEAD,
                    help="WPA classifier: 2-layer MLP on the text token (default, as in the paper) or linear (the "
                         "default before scale-v1); mlp_patch (MLP on [text; covered patches]) leaks the label "
                         "through the [MASK] patch embedding: probes only")
    ap.add_argument("--mim_weight", type=float, default=1.0,
                    help="weight of the MIM loss in training (0: MLM + WPA only; images are still masked for WPA)")
    ap.add_argument("--freeze_layers", type=int, default=0,
                    help="during joint training keep the embeddings and the bottom N transformer layers frozen")
    ap.add_argument("--random_init", action="store_true",
                    help="control: use --model's config but random weights (pre-training from scratch)")
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many evaluations without a lower validation loss (0: never)")
    ap.add_argument("--save_last", action="store_true", help="also save the final step to last_backbone/")
    ap.add_argument("--state_every", type=int, default=0,
                    help="save model, optimiser and scheduler state to output_dir/train_state.pt every N steps "
                         "(0: never); removed when the run finishes")
    ap.add_argument("--resume", action="store_true",
                    help="continue from output_dir/train_state.pt if it exists. With --corpus idl the page stream "
                         "continues where it stopped: the workers re-read and skip the pages of the batches already "
                         "used (same seed, --batch_size and --num_workers required), so no page is replayed. Masks "
                         "after a resume come from a new random stream, seeded with (seed, step). With --corpus hf "
                         "the epoch order is reshuffled instead")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tags", nargs="*", default=[], help="free-form tags for experiments/registry.jsonl")
    return ap


def main():
    a = build_parser().parse_args()

    tags = list(a.tags)
    if a.head_warmup_steps and a.max_steps == 0:
        tags.append("probe")
    with RunTracker("pretrain", a, output_dir=a.output_dir, tags=tags) as run:
        train(a, run)


def train(a, run):
    torch.manual_seed(a.seed)
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_path = out_dir / "train_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False) if a.resume and state_path.exists() else None
    # The page order is a function of the seed only. On resume the IDL workers skip the batches already used
    # (step x grad_accum micro-batches), so the stream continues exactly where the interrupted run stopped;
    # the masks of the remaining batches are drawn from a new stream seeded with (seed, step).
    a.data_seed, a.resume_batches, mask_seed = a.seed, 0, a.seed
    if state is not None:
        if state.get("batch_size", a.batch_size) != a.batch_size or state.get("num_workers", a.num_workers) != a.num_workers:
            raise SystemExit("--resume needs the --batch_size and --num_workers of the interrupted run "
                             f"({state.get('batch_size')}, {state.get('num_workers')})")
        a.resume_batches = state["step"] * a.grad_accum
        mask_seed = (a.seed, state["step"])
        print(f"resuming from step {state['step']}: skipping {a.resume_batches} batches of the page stream", flush=True)
    processor = LayoutLMv3Processor.from_pretrained(a.model, apply_ocr=False)

    t0 = time.time()
    train, val, fit_images, info = (build_idl_corpus if a.corpus == "idl" else build_hf_corpus)(a, processor)

    if a.image_tokenizer == "dalle":
        img_tok = DalleTokenizer.from_pretrained(a.dalle_encoder).to(device)
    else:
        img_tok = KMeansPatchTokenizer.fit(fit_images, vocab_size=a.image_vocab_size, seed=a.seed)
        img_tok.save(out_dir / "image_tokenizer.npz")
    info.update(image_tokenizer=a.image_tokenizer, image_vocab_size=img_tok.vocab_size,
                setup_minutes=round((time.time() - t0) / 60, 1), args=vars(a))
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    run.config.update(
        model=a.model, init="random" if a.random_init else "pretrained", wpa_head=a.wpa_head,
        image_tokenizer=a.image_tokenizer, image_tokenizer_path=a.dalle_encoder if a.image_tokenizer == "dalle"
        else str(out_dir / "image_tokenizer.npz"), image_vocab_size=img_tok.vocab_size, corpus=a.corpus,
        train_shards=a.idl_shards if a.corpus == "idl" else None, val_shards=a.idl_val_shards if a.corpus == "idl" else None,
        dataset=a.dataset if a.corpus == "hf" else None, seed=a.seed, head_warmup_steps=a.head_warmup_steps,
        head_lr=a.head_lr, max_steps=a.max_steps, lr=a.lr, layer_decay=a.layer_decay,
        effective_batch=a.batch_size * a.grad_accum, mim_weight=a.mim_weight, freeze_layers=a.freeze_layers,
        corpus_info={k: v for k, v in info.items() if k not in ("args", "val_keys")})

    model = LayoutLMv3ForPreTraining.from_backbone(a.model, img_tok.vocab_size, random_init=a.random_init,
                                                 wpa_head=a.wpa_head).to(device)
    info["loading_info"] = model.loading_info
    run.config["loading_info"] = model.loading_info
    print(json.dumps({"loading_info": model.loading_info}), flush=True)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=1))
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    collate = PretrainingCollator(processor.tokenizer, None, seed=mask_seed)    # seeded per DataLoader worker
    train_loader = DataLoader(train, batch_size=a.batch_size, drop_last=True, collate_fn=collate,
                              shuffle=a.corpus == "hf", num_workers=a.num_workers,
                              persistent_workers=a.num_workers > 0, pin_memory=device == "cuda",
                              prefetch_factor=4 if a.num_workers > 0 else None)
    # validation masks are drawn once, so every evaluation (and the best-checkpoint choice) uses the same ones
    val_collate = PretrainingCollator(processor.tokenizer, None, seed=0)
    val_batches = [val_collate(val[i:i + a.batch_size]) for i in range(0, len(val), a.batch_size)]

    def joint_optimiser():
        model.set_encoder_trainable(True)
        if a.freeze_layers:
            model.freeze_bottom(a.freeze_layers)
        groups = model.layerwise_param_groups(a.lr, a.layer_decay, weight_decay=0.01,
                                              new_lr=a.new_params_lr)
        groups = [g for g in groups if g["params"][0].requires_grad]       # groups are uniform per depth
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=ADAM_BETAS)
        return opt, get_cosine_schedule_with_warmup(opt, int(a.warmup_ratio * a.max_steps), a.max_steps)

    warmup = a.head_warmup_steps
    if warmup:  # phase 1: heads only, encoder frozen
        model.set_encoder_trainable(False)
        new = {id(p) for p in model.new_parameters().values()}
        groups = [g for g in model.layerwise_param_groups(a.head_lr, weight_decay=0.01) if id(g["params"][0]) in new]
        opt = torch.optim.AdamW(groups, lr=a.head_lr, betas=ADAM_BETAS)
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * warmup), warmup)
        enc_before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                      if n not in model.new_parameters()}
    else:
        opt, sched = joint_optimiser()
    total_steps = warmup + a.max_steps

    reference = model.checkpoint_state() if not a.random_init else None
    run.outputs.update(log=str(out_dir / "log.json"), run_info=str(out_dir / "run_info.json"))
    best, bad_evals, step, micro, t0 = float("inf"), 0, 0, 0, time.time()
    if state is None:
        log = [{"step": 0, "phase": "head_warmup" if warmup else "joint",
                **evaluate(model, val_batches, device, img_tok)}]
        print(json.dumps(log[-1]), flush=True)
        run.log({f"val/{k}": v for k, v in log[-1].items() if k != "step"}, step=0)
    else:
        model.load_state_dict(state["model"])
        step, best, bad_evals, log = state["step"], state["best"], state["bad_evals"], state["log"]
        t0 -= 60 * state["minutes"]
        if warmup and step >= warmup:
            opt, sched = joint_optimiser()
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        run.metrics["resumed_from_steps"] = state.get("resumed_from_steps", []) + [step]
        del state
    run.metrics["step0"] = log[0]
    done = step >= total_steps
    while not done:
        for batch in train_loader:
            batch = to_device(batch, device, img_tok)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(**batch)
            loss = out.loss if a.mim_weight == 1.0 else out.mlm_loss + a.mim_weight * out.mim_loss + out.wpa_loss
            (loss / a.grad_accum).backward()
            micro += 1
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            in_warmup = step <= warmup
            if step % a.eval_every == 0 or step == warmup or step == total_steps:
                ev = evaluate(model, val_batches, device, img_tok)
                if step == warmup:
                    ev["encoder_max_abs_change"] = max(
                        (p.detach().cpu() - enc_before[n]).abs().max().item()
                        for n, p in model.named_parameters() if n in enc_before)
                ev["sel_loss"] = round(ev["mlm_loss"] + a.mim_weight * ev["mim_loss"] + ev["wpa_loss"], 4)
                improved = ev["sel_loss"] < best
                if improved:
                    best, bad_evals = ev["sel_loss"], 0
                    save_backbone(model, processor, out_dir / "backbone")
                    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("layoutlmv3.")},
                               out_dir / "pretraining_heads.pt")
                else:
                    bad_evals += 1
                log.append({"step": step, "phase": "head_warmup" if in_warmup else "joint",
                            "train_loss": round(out.loss.item(), 4), **ev, "best": improved,
                            "pages_seen": step * a.batch_size * a.grad_accum,
                            "minutes": round((time.time() - t0) / 60, 1)})
                print(json.dumps(log[-1]), flush=True)
                run.log({f"val/{k}": v for k, v in log[-1].items() if k != "step"}, step=step)
                (out_dir / "log.json").write_text(json.dumps(log, indent=1))
                if step == warmup:
                    save_backbone(model, processor, out_dir / "warmup_backbone")
                    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("layoutlmv3.")}
                               | {"layoutlmv3.image_mask_token": model.layoutlmv3.image_mask_token.detach()},
                               out_dir / "warmup_heads.pt")
                    best, bad_evals = float("inf"), 0       # select the checkpoint among joint steps only
                if a.patience and not in_warmup and bad_evals >= a.patience:
                    print(f"early stop: no improvement in {a.patience} evaluations", flush=True)
                    done = True
            if step == warmup:                       # phase 2: joint training
                opt, sched = joint_optimiser()
                micro = 0
            if a.state_every and step % a.state_every == 0 and step < total_steps and not done:
                save_train_state(state_path, model=model.state_dict(), optimizer=opt.state_dict(),
                                 scheduler=sched.state_dict(), step=step, best=best, bad_evals=bad_evals, log=log,
                                 minutes=round((time.time() - t0) / 60, 2), batch_size=a.batch_size,
                                 num_workers=a.num_workers,
                                 resumed_from_steps=run.metrics.get("resumed_from_steps", []))
            if step >= total_steps:
                done = True
            if done:
                break

    if a.save_last:
        save_backbone(model, processor, out_dir / "last_backbone")
    state_path.unlink(missing_ok=True)
    if reference is not None:  # drift of the final weights from the starting checkpoint
        drift = {"final_step": step, **model.encoder_drift(reference)}
        (out_dir / "drift.json").write_text(json.dumps(drift, indent=1))
        run.metrics["encoder_drift_final"] = drift
        print(json.dumps({"encoder_drift": drift}), flush=True)
    joint = [r for r in log[1:] if r["phase"] == "joint"]
    best_step = min(joint, key=lambda r: r["sel_loss"])["step"] if joint else 0
    after_warmup = [r for r in log if r["step"] == warmup and r["phase"] == "head_warmup" and warmup]
    run.metrics.update(after_head_warmup=after_warmup[0] if after_warmup else None, final=log[-1],
                       best_joint=min(joint, key=lambda r: r["sel_loss"]) if joint else None,
                       steps_done=step, wall_minutes=round((time.time() - t0) / 60, 1))
    run.outputs.update(backbone=str(out_dir / "backbone") if joint else None,
                       warmup_backbone=str(out_dir / "warmup_backbone") if warmup else None,
                       last_backbone=str(out_dir / "last_backbone") if a.save_last else None)
    print(f"best validation loss {best:.4f} at step {best_step}; saved to {out_dir / 'backbone'}", flush=True)


if __name__ == "__main__":
    main()
