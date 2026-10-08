"""End-to-end: a tiny model trains on synthetic data, writes its outputs, and resumes safely."""

import json

import numpy as np
import pytest

from thinkpad.train import main


@pytest.fixture
def data_dir(tmp_path):
    """A predictable token stream (a repeating 0..49 cycle), so loss must fall fast."""
    d = tmp_path / "data"
    d.mkdir()
    for split, n in [("train", 20_000), ("validation", 2_000), ("test", 2_000)]:
        (np.arange(n) % 50).astype(np.uint16).tofile(d / f"{split}.bin")
    return d


def run(data_dir, out_dir, arch="thinkpad-tiny", **overrides):
    args = {
        "arch": arch,
        "data_dir": data_dir,
        "out_dir": out_dir,
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 8,
        "max_iters": 40,
        "eval_interval": 20,
        "eval_iters": 2,
        "log_interval": 10,
        "learning_rate": 3e-3,
        "min_lr": 3e-4,
        "warmup_iters": 5,
        "sample_tokens": 0,
        **overrides,
    }
    argv = ["--skip_prepare"]
    for k, v in args.items():
        argv += [f"--{k}", str(v)]
    main(argv)


def evals(out_dir):
    lines = (out_dir / "metrics.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    return [r for r in rows if r["kind"] == "eval"]


@pytest.mark.parametrize("arch", ["thinkpad-tiny", "baseline-tiny"])
def test_training_learns_and_writes_outputs(data_dir, tmp_path, arch):
    out = tmp_path / arch
    run(data_dir, out, arch=arch)
    for name in ("config.json", "metrics.jsonl", "checkpoint.pt", "best_model.pt", "results.json"):
        assert (out / name).exists(), name
    rows = evals(out)
    assert [r["iter"] for r in rows] == [0, 20, 40]
    assert rows[-1]["val_loss"] < rows[0]["val_loss"] - 3  # from ~10.8 down a long way
    results = json.loads((out / "results.json").read_text())
    assert results["test_loss"] < 6


def test_resume_continues_without_repeating_evals(data_dir, tmp_path):
    out = tmp_path / "run"
    run(data_dir, out, max_iters=20)
    run(data_dir, out, max_iters=40)
    assert [r["iter"] for r in evals(out)] == [0, 20, 40]


def test_resume_with_different_model_is_refused(data_dir, tmp_path):
    out = tmp_path / "run"
    run(data_dir, out, max_iters=20)
    with pytest.raises(SystemExit, match="different model config"):
        run(data_dir, out, max_iters=20, n_layer=3)


def test_flops_budget_sets_step_count(data_dir, tmp_path):
    out = tmp_path / "run"
    run(data_dir, out, max_iters=1, eval_interval=1000)
    per_step = json.loads((out / "config.json").read_text())["flops_per_iter"]
    out2 = tmp_path / "budget"
    run(data_dir, out2, flops_budget=per_step * 30 + 1, eval_interval=1000)
    assert json.loads((out2 / "config.json").read_text())["max_iters"] == 30
    assert evals(out2)[-1]["iter"] == 30
