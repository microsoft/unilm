"""Print the runs in experiments/registry.jsonl as a table.

    python scripts/list_runs.py                       # all runs
    python scripts/list_runs.py --tag idl_warmup      # runs with this tag (repeatable: all must match)
    python scripts/list_runs.py --kind pretrain --status completed
    python scripts/list_runs.py --json <run_id>       # the full record of one run
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from layoutlmv3_pretraining.tracking import read_registry  # noqa: E402


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def wpa_head_of(r):
    """WPA head of a pre-training record. The default changed from linear to mlp (scale-v1); every record written
    since --wpa_head existed stores the head explicitly, and the only ones without it (backfilled runs from before
    the flag) all used the linear head, so a missing value means linear, never the current default."""
    c, a = r.get("config") or {}, r.get("args") or {}
    return c.get("wpa_head") or a.get("wpa_head") or "linear"


def row(r):
    c, m, a = r.get("config") or {}, r.get("metrics") or {}, r.get("args") or {}
    date = (r.get("end_utc") or r.get("start_utc") or "")[:16].replace("T", " ") or "-"
    if r["kind"] == "pretrain":
        warm = c.get("head_warmup_steps") or 0
        steps = c.get("max_steps", a.get("max_steps"))
        if warm and steps == 0:   # probe: only the head lr matters
            hp = f"head-only {warm} steps, head_lr={fmt(c.get('head_lr'), 4)}"
        else:
            hp = f"warm={warm} steps={steps} lr={fmt(c.get('lr', a.get('lr')), 6)}"
        if c.get("layer_decay") not in (None, 1.0):
            hp += f" ld={c['layer_decay']}"
        best = m.get("best_joint") or m.get("after_head_warmup") or m.get("final") or {}
        head = (f"mlm {fmt(best.get('mlm_acc'))} mim {fmt(best.get('mim_acc'))} wpa {fmt(best.get('wpa_acc'))}"
                + (f" auc {fmt(best.get('wpa_auc'))}" if best.get("wpa_auc") is not None else ""))
        tok = c.get("image_tokenizer", "-")
        if tok == "dalle" and "dvae_doc" in str(c.get("image_tokenizer_path")):
            tok = "dalle_doc"
        variant = f"wpa={wpa_head_of(r)} mim={tok}"
        if c.get("mim_weight", 1.0) != 1.0:
            variant += f" mim_w={c['mim_weight']}"
        if c.get("freeze_layers"):
            variant += f" freeze={c['freeze_layers']}"
        init, model = c.get("init", "-"), c.get("model", a.get("model", "-"))
    elif r["kind"].startswith("finetune_") and r["kind"] != "finetune_dvae":
        hp = f"steps={c.get('steps')} lr={fmt(c.get('lr'), 6)} bs={c.get('batch_size')} seeds={len(c.get('seeds') or [])}"
        head = f"F1 {fmt(m.get('mean'))} ± {fmt(m.get('std'))}"
        variant, init, model = r["kind"].removeprefix("finetune_"), "-", c.get("model", "-")
    else:  # finetune_dvae and anything else
        hp = f"steps={a.get('steps')} lr={fmt(a.get('lr'), 6)} bs={a.get('batch_size')}"
        b, af = m.get("before") or {}, m.get("after") or {}
        head = f"PSNR {fmt(b.get('psnr'), 2)} -> {fmt(af.get('psnr'), 2)}" if b else "-"
        variant, init, model = r["kind"], "-", "dall-e dVAE"
    vtag = [t for t in r.get("tags", []) if t.startswith("variant-")]
    if vtag:
        variant = f"[{vtag[0][8:]}] {variant}"
    status = r["status"] + ("*" if r.get("backfilled") else "")
    return [r["run_id"], date, status, r["kind"], str(model).replace("microsoft/", ""), init, variant, hp, head]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry")
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--kind")
    ap.add_argument("--status")
    ap.add_argument("--json", metavar="RUN_ID")
    a = ap.parse_args()
    runs = read_registry(a.registry)
    if a.json:
        print(json.dumps(next(r for r in runs if r["run_id"] == a.json), indent=1))
        return
    runs = [r for r in runs if all(t in r.get("tags", []) for t in a.tag)
            and (not a.kind or r["kind"] == a.kind) and (not a.status or r["status"] == a.status)]
    header = ["run_id", "date (UTC)", "status", "kind", "model", "init", "variant", "hyperparameters", "headline"]
    rows = [header] + [row(r) for r in runs]
    widths = [max(len(x[i]) for x in rows) for i in range(len(header))]
    for i, x in enumerate(rows):
        print("  ".join(v.ljust(w) for v, w in zip(x, widths)).rstrip())
        if i == 0:
            print("  ".join("-" * w for w in widths))
    print(f"\n{len(runs)} runs (* = backfilled from results files; unknown fields are null)")


if __name__ == "__main__":
    main()
