"""Step switches: every ablation variant builds, is causal, trains every parameter it has,
and has its FLOPs counted exactly."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from thinkpad import MODEL_PRESETS, build_model
from thinkpad.ablate import VARIANTS
from thinkpad.ablate import main as ablate_main
from thinkpad.config import ModelConfig, layer_set
from thinkpad.flops import analytic_train_flops, measure_train_flops

BASE = replace(MODEL_PRESETS["thinkpad-tiny"], n_layer=4, dropout=0.0)
THINKPAD_VARIANTS = [v for v in VARIANTS if not v.baseline]


def cfg_for(v) -> ModelConfig:
    return replace(BASE, **v.steps)


def test_layer_specs():
    assert layer_set("all", 4) == {0, 1, 2, 3}
    assert layer_set("none", 4) == set()
    assert layer_set("even", 5) == {0, 2, 4}
    assert layer_set("odd", 5) == {1, 3}
    assert layer_set("first:half", 8) == {0, 1, 2, 3}
    assert layer_set("last:2", 8) == {6, 7}
    assert layer_set("0, 3", 4) == {0, 3}
    for bad in ("first:9", "7", "sometimes"):
        with pytest.raises(ValueError):
            layer_set(bad, 4)


def test_p_must_read_x_somewhere():
    with pytest.raises(ValueError, match="never reads"):
        replace(BASE, p_read1="none", p_read2="none", gate_p="none")


def test_full_variant_is_the_default_model():
    assert cfg_for(VARIANTS[0]) == BASE
    assert BASE.ablated_steps() == {}


@pytest.mark.parametrize("v", THINKPAD_VARIANTS, ids=lambda v: v.name)
def test_variant_flops_exact(v):
    model = build_model(cfg_for(v))
    assert analytic_train_flops(model, 2) == measure_train_flops(model, 2)


@pytest.mark.parametrize("v", THINKPAD_VARIANTS, ids=lambda v: v.name)
def test_variant_is_causal(v):
    torch.manual_seed(0)
    cfg = cfg_for(v)
    model = build_model(cfg).eval()
    idx = torch.randint(cfg.vocab_size, (2, cfg.block_size))
    k = cfg.block_size // 2
    changed = idx.clone()
    changed[:, k:] = torch.randint(cfg.vocab_size, changed[:, k:].shape)
    with torch.no_grad():
        a, b = model(idx)[0], model(changed)[0]
    torch.testing.assert_close(a[:, :k], b[:, :k])


@pytest.mark.parametrize("v", THINKPAD_VARIANTS, ids=lambda v: v.name)
def test_variant_trains_all_its_parameters(v):
    """Only LayerNorm scales that normalize a still-all-zero p may stay untrained."""
    torch.manual_seed(0)
    cfg = cfg_for(v)
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        idx = torch.randint(cfg.vocab_size, (2, cfg.block_size))
        opt.zero_grad()
        model(idx, torch.randint(cfg.vocab_size, idx.shape))[1].backward()
        opt.step()
    modules = dict(model.named_modules())
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    for name in dead:
        owner, leaf = name.rsplit(".", 1)
        assert leaf == "weight" and isinstance(modules[owner], torch.nn.LayerNorm), name


def test_switched_off_steps_are_not_built():
    model = build_model(replace(BASE, p_read2="none", bypass="none"))
    names = {n.split(".")[2] for n, _ in model.named_parameters() if n.startswith("blocks.")}
    assert not names & {"mha_p2", "ln_p2_p", "ln_p2_x", "mha_bypass", "ln_by_x", "ln_by_p"}
    assert model.blocks[0].ffwd_x.net[0].in_features == BASE.n_embd  # no bypass input


def test_ablation_runner_end_to_end(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for split, n in [("train", 20_000), ("validation", 2_000), ("test", 2_000)]:
        (np.arange(n) % 50).astype(np.uint16).tofile(d / f"{split}.bin")
    out = tmp_path / "abl"
    ablate_main([
        "--model", "thinkpad-tiny", "--baseline", "baseline-tiny", "--dataset", "wikitext103",
        "--data_dir", str(d), "--out_root", str(out), "--steps", "40", "--batch_size", "2",
        "--only", "full", "one_read_no_bypass", "gpt_baseline",
        "--", "--device", "cpu", "--eval_interval", "1000", "--eval_iters", "1",
        "--sample_tokens", "0", "--skip_prepare",
    ])  # fmt: skip
    summary = (out / "summary.md").read_text()
    for name in ("full", "one_read_no_bypass", "gpt_baseline"):
        assert name in summary
    # a cheaper variant gets more steps for the same compute
    import json

    rows = {r["variant"]: r for r in json.loads((out / "summary.json").read_text())}
    assert rows["full"]["steps"] == 40
    assert rows["one_read_no_bypass"]["steps"] > 40


def test_p_self_attention_runs_before_the_gate():
    """sa_p sits between p's reads and the step-4 gate, so in the same block it already
    changes what the gate passes into x."""
    torch.manual_seed(0)
    cfg = replace(BASE, sa_p="0")
    model = build_model(cfg).eval()
    block = model.blocks[0]
    assert block.use_sa_p and block.use_gate_x
    x = torch.randn(2, 8, cfg.n_embd)
    with torch.no_grad():
        x_out, _ = block(x, torch.zeros_like(x))
        block.sa_p.proj.weight.mul_(5.0)
        x_out2, _ = block(x, torch.zeros_like(x))
    assert not torch.allclose(x_out, x_out2)


def test_p_self_attention_off_by_default():
    assert BASE.sa_p == "none"
    assert not any(b.use_sa_p for b in build_model(BASE).blocks)
    assert replace(BASE, sa_p="all").ablated_steps() == {"sa_p": "all"}
