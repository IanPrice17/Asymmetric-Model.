import pytest
import torch

from thinkpad import MODEL_PRESETS, build_model, count_params
from thinkpad.config import ModelConfig


def tiny(arch: str) -> ModelConfig:
    return ModelConfig.from_dict({**MODEL_PRESETS[f"{arch}-tiny"].to_dict(), "dropout": 0.0})


ARCHS = ["thinkpad", "baseline"]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_shapes_and_loss(arch):
    cfg = tiny(arch)
    model = build_model(cfg)
    idx = torch.randint(cfg.vocab_size, (3, 17))
    logits, loss = model(idx)
    assert logits.shape == (3, 17, cfg.vocab_size)
    assert loss is None
    _, loss = model(idx, torch.randint(cfg.vocab_size, (3, 17)))
    # A freshly initialised model should be close to uniform over the vocabulary.
    assert abs(loss.item() - torch.log(torch.tensor(cfg.vocab_size)).item()) < 0.5


@pytest.mark.parametrize("arch", ARCHS)
def test_is_causal(arch):
    """Changing token k must not change predictions at positions < k (both streams)."""
    cfg = tiny(arch)
    model = build_model(cfg).eval()
    idx = torch.randint(cfg.vocab_size, (2, cfg.block_size))
    k = cfg.block_size // 2
    changed = idx.clone()
    changed[:, k:] = torch.randint(cfg.vocab_size, changed[:, k:].shape)
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    torch.testing.assert_close(a[:, :k], b[:, :k])
    assert not torch.allclose(a[:, k:], b[:, k:])


@pytest.mark.parametrize("arch", ARCHS)
def test_every_parameter_is_trained(arch):
    """No parameter may be cut off from the loss (this caught the old last-block x layers).

    One optimizer step is taken first: p starts at zero, so in block 0 the first
    p->x attention has an all-zero query until LayerNorm's bias moves, and its
    query/key weights get their first gradient on the second step.
    """
    cfg = tiny(arch)
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        idx = torch.randint(cfg.vocab_size, (2, cfg.block_size))
        opt.zero_grad()
        model(idx, torch.randint(cfg.vocab_size, idx.shape))[1].backward()
        opt.step()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    # Block 0's ln_p1_p normalizes p == 0, whose normalized value is always 0, so its
    # scale can never get a gradient. Known and harmless (n_embd parameters).
    expected = ["blocks.0.ln_p1_p.weight"] if arch == "thinkpad" else []
    assert dead == expected


@pytest.mark.parametrize("arch", ARCHS)
def test_weight_tying(arch):
    model = build_model(tiny(arch))
    assert model.lm_head.weight is model.token_embedding_table.weight


@pytest.mark.parametrize("arch", ARCHS)
def test_generate(arch):
    cfg = tiny(arch)
    model = build_model(cfg).train()
    prompt = torch.zeros((1, 5), dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=cfg.block_size + 3, top_k=10)
    assert out.shape == (1, 5 + cfg.block_size + 3)  # longer than context: must crop, not crash
    assert torch.equal(out[:, :5], prompt)
    assert model.training  # mode restored


def test_sequence_longer_than_context_is_rejected():
    cfg = tiny("thinkpad")
    with pytest.raises(ValueError, match="block_size"):
        build_model(cfg)(torch.zeros((1, cfg.block_size + 1), dtype=torch.long))


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=100, n_head=6)
    with pytest.raises(ValueError):
        ModelConfig(arch="mamba")


def test_preset_sizes():
    """Regression guard: the experiment presets' sizes are part of the results."""
    assert count_params(build_model(MODEL_PRESETS["thinkpad"])) == 64_092_288
    assert count_params(build_model(MODEL_PRESETS["baseline"])) == 66_915_840
