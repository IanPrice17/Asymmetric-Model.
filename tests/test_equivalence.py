"""The refactored models compute exactly what the original Colab code computed.

``legacy_thinkpad`` / ``legacy_baseline`` hold the original architecture code
unchanged. Weights are copied from an original model into the new one through
``load_state_dict_compat`` (the same path used to load old checkpoints), and
the outputs and gradients must match.
"""

import legacy_baseline
import legacy_thinkpad
import pytest
import torch

from thinkpad import MODEL_PRESETS, build_model, load_state_dict_compat
from thinkpad.config import ModelConfig

LEGACY = {"thinkpad": legacy_thinkpad, "baseline": legacy_baseline}


def tiny(arch: str) -> ModelConfig:
    return ModelConfig.from_dict({**MODEL_PRESETS[f"{arch}-tiny"].to_dict(), "dropout": 0.0})


@pytest.mark.parametrize("arch", ["thinkpad", "baseline"])
@pytest.mark.parametrize("seq_len", [64, 23])
def test_same_outputs_and_gradients(arch, seq_len):
    torch.manual_seed(0)
    cfg = tiny(arch)
    old = LEGACY[arch].build(cfg)
    new = build_model(cfg)
    dropped = load_state_dict_compat(new, old.state_dict())

    idx = torch.randint(cfg.vocab_size, (2, seq_len))
    old_logits, old_loss = old(idx, idx)
    new_logits, new_loss = new(idx, idx)
    # (the original returned logits flattened to (B*T, V) whenever targets were given)
    torch.testing.assert_close(new_logits.view_as(old_logits), old_logits, rtol=0, atol=1e-6)
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=1e-6)

    old_loss.backward()
    new_loss.backward()
    old_params = dict(old.named_parameters())
    for name, p in new.named_parameters():
        torch.testing.assert_close(p.grad, old_params[name].grad, rtol=1e-5, atol=1e-7)

    # Everything that was dropped really was dead weight in the original model.
    for name in dropped:
        assert old_params[name].grad is None, name


def test_dropped_keys_are_exactly_the_dead_ones():
    cfg = tiny("thinkpad")
    new = build_model(cfg)
    dropped = load_state_dict_compat(new, legacy_thinkpad.build(cfg).state_dict())
    last = f"blocks.{cfg.n_layer - 1}."
    assert all(k.startswith((last, "ln_f.")) for k in dropped)
    assert {k.removeprefix(last).split(".")[0] for k in dropped if k.startswith(last)} == {
        "gate_x",
        "ln_x_post",
        "mha_bypass",
        "ln_by_x",
        "ln_by_p",
        "ffwd_x",
        "ln_x_ffn",
    }


def test_missing_keys_still_fail():
    cfg = tiny("baseline")
    state = build_model(cfg).state_dict()
    del state["blocks.0.ln1.weight"]
    with pytest.raises(RuntimeError, match="Missing"):
        load_state_dict_compat(build_model(cfg), state)


def test_original_colab_checkpoint_loads_for_sampling(tmp_path):
    """Checkpoints saved by the original scripts ('model_state' + 'config') still load."""
    from thinkpad.sample import load_model

    cfg = tiny("thinkpad")
    old = legacy_thinkpad.build(cfg)
    path = str(tmp_path / "best_model.pt")
    torch.save(
        {
            "iter": 0,
            "model_state": old.state_dict(),
            "best_val_loss": 1.0,
            "config": {
                "n_embd": cfg.n_embd,
                "n_head": cfg.n_head,
                "n_layer": cfg.n_layer,
                "block_size": cfg.block_size,
                "vocab_size": cfg.vocab_size,
            },
        },
        path,
    )
    with pytest.raises(SystemExit, match="--arch"):
        load_model(path)
    model = load_model(path, arch="thinkpad")
    idx = torch.randint(cfg.vocab_size, (1, 10))
    old.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(idx)[0], old(idx)[0], rtol=0, atol=1e-6)
