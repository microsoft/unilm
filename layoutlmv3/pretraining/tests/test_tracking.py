import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from layoutlmv3_pretraining.tracking import RunTracker, append_record, git_info, read_registry, to_jsonable


def test_completed_run_is_recorded_with_copy(tmp_path):
    reg, out = tmp_path / "registry.jsonl", tmp_path / "run"
    args = argparse.Namespace(lr=1e-5, seeds=[1, 2], output_dir=str(out))
    with RunTracker("pretrain", args, output_dir=out, tags=["t1"], registry=reg, argv=["x.py", "--lr", "1e-5"]) as run:
        run.config["model"] = "m"
        run.metrics["f1"] = np.float32(0.5)
        run.outputs["backbone"] = out / "backbone"
    (rec,) = read_registry(reg)
    assert rec["status"] == "completed" and rec["kind"] == "pretrain" and rec["tags"] == ["t1"]
    assert rec["args"] == {"lr": 1e-5, "seeds": [1, 2], "output_dir": str(out)}
    assert rec["config"] == {"model": "m"} and rec["metrics"]["f1"] == pytest.approx(0.5)
    assert rec["argv"] == ["x.py", "--lr", "1e-5"] and rec["error"] is None and not rec["backfilled"]
    assert rec["outputs"]["output_dir"] == str(out) and rec["outputs"]["backbone"].endswith("backbone")
    for key in ("run_id", "start_utc", "end_utc", "git", "environment", "command"):
        assert key in rec
    assert {"python", "cuda_visible_devices"} <= set(rec["environment"])
    assert json.loads((out / "run_record.json").read_text()) == rec


def test_failed_run_is_recorded_and_exception_propagates(tmp_path):
    reg = tmp_path / "registry.jsonl"
    with pytest.raises(RuntimeError, match="boom"):
        with RunTracker("finetune_funsd", {"a": 1}, registry=reg, argv=[]) as run:
            run.metrics["partial"] = 1
            raise RuntimeError("boom")
    (rec,) = read_registry(reg)
    assert rec["status"] == "failed" and rec["error"] == "RuntimeError: boom" and rec["metrics"] == {"partial": 1}


def _append(args):
    reg, i = args
    append_record({"run_id": i, "payload": "x" * 20000}, reg)


def test_concurrent_appends_never_interleave(tmp_path):
    reg = tmp_path / "registry.jsonl"
    with ProcessPoolExecutor(4) as ex:
        list(ex.map(_append, [(str(reg), i) for i in range(40)]))
    assert sorted(r["run_id"] for r in read_registry(reg)) == list(range(40))


def test_git_info_detects_dirty_tree(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "f.txt").write_text("a"); git("add", "f.txt"); git("commit", "-qm", "c")
    clean = git_info(tmp_path)
    assert len(clean["commit"]) == 40 and clean["dirty"] is False and clean["diff_sha256"] is None
    (tmp_path / "f.txt").write_text("b"); (tmp_path / "new.txt").write_text("n")
    dirty = git_info(tmp_path)
    assert dirty["dirty"] and len(dirty["diff_sha256"]) == 64 and dirty["untracked"] == ["new.txt"]
    (tmp_path / "f.txt").write_text("c")
    assert git_info(tmp_path)["diff_sha256"] != dirty["diff_sha256"]
    assert git_info(Path("/"))["commit"] is None                             # not a repository


def test_to_jsonable():
    import torch
    out = to_jsonable({"p": Path("/a"), "t": torch.tensor(2.5), "n": np.int64(3), "s": {1}, 1: (1, 2)})
    assert out == {"p": "/a", "t": 2.5, "n": 3, "s": [1], "1": [1, 2]}
    json.dumps(out)


def test_list_runs_reports_the_wpa_head_of_old_and_new_records():
    """The default WPA head changed from linear to mlp; records without the field predate the flag (linear)."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "list_runs.py"
    spec = importlib.util.spec_from_file_location("list_runs", path)
    list_runs = importlib.util.module_from_spec(spec); spec.loader.exec_module(list_runs)
    base = {"run_id": "r", "kind": "pretrain", "status": "completed", "tags": []}
    assert list_runs.wpa_head_of({**base, "config": None}) == "linear"                       # backfilled, no config
    assert list_runs.wpa_head_of({**base, "config": {"model": "m"}}) == "linear"
    assert list_runs.wpa_head_of({**base, "config": {"wpa_head": "linear"}}) == "linear"
    assert list_runs.wpa_head_of({**base, "config": {"wpa_head": "mlp"}}) == "mlp"
    assert "wpa=mlp " in " ".join(list_runs.row({**base, "config": {"wpa_head": "mlp", "image_tokenizer": "dalle"}}))
    assert "wpa=linear " in " ".join(list_runs.row({**base, "config": None}))


def test_run_tracker_logs_to_wandb_when_project_is_set(tmp_path, monkeypatch):
    # W&B logging is opt-in through LAYOUTLMV3_WANDB_PROJECT; offline mode needs no network or account.
    pytest = __import__("pytest")
    pytest.importorskip("wandb")
    from layoutlmv3_pretraining.tracking import RunTracker, read_registry
    monkeypatch.setenv("LAYOUTLMV3_WANDB_PROJECT", "layoutlmv3-test")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    reg = tmp_path / "registry.jsonl"
    with RunTracker("unit", {"lr": 1e-5}, tags=["t"], registry=reg, argv=["x"]) as run:
        assert run.wandb_run is not None
        run.log({"val/loss": 1.5, "phase": "joint", "best": True}, step=10)
        run.metrics["final"] = 0.9
    rec = read_registry(reg)[-1]
    assert rec["status"] == "completed" and "wandb_url" in rec["outputs"]


def test_run_tracker_without_wandb_project_is_unchanged(tmp_path, monkeypatch):
    from layoutlmv3_pretraining.tracking import RunTracker, read_registry
    monkeypatch.delenv("LAYOUTLMV3_WANDB_PROJECT", raising=False)
    reg = tmp_path / "registry.jsonl"
    with RunTracker("unit", {}, registry=reg, argv=["x"]) as run:
        run.log({"val/loss": 1.0}, step=1)  # no-op
        assert run.wandb_run is None
    assert "wandb_url" not in read_registry(reg)[-1]["outputs"]
