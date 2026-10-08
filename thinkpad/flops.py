"""Training-compute accounting.

Convention (the usual one in scaling-law work):

* Only matrix multiplications are counted; one multiply-add = 2 FLOPs.
  LayerNorm, GELU, softmax, embedding lookups and the optimizer are ignored.
* Attention is counted as dense (T x T) even though it is causal, matching
  ``torch.utils.flop_counter``.
* Backward pass = 2x forward, so one training step = 3x forward.

``analytic_train_flops`` gives the closed form; ``measure_train_flops`` counts
the FLOPs PyTorch actually executes for one forward + backward pass. The tests
check that the two agree exactly.

Run ``python -m thinkpad.flops`` to print the numbers for both presets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from .config import MODEL_PRESETS, ModelConfig
from .model import FusedSelfAttention, MultiHeadAttention, build_model, count_params


def _linear_weights(model: nn.Module) -> int:
    """Weights of every nn.Linear, counting a tied lm_head (it still does a matmul)."""
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))


def _attention_ops(model: nn.Module) -> int:
    return sum(isinstance(m, (MultiHeadAttention, FusedSelfAttention)) for m in model.modules())


def analytic_forward_flops_per_token(model: nn.Module, seq_len: int) -> int:
    """2 FLOPs per Linear weight, plus QK^T and AV (2 * T * d each) per attention op."""
    d = model.cfg.n_embd
    return 2 * _linear_weights(model) + _attention_ops(model) * 4 * seq_len * d


def analytic_train_flops(model: nn.Module, batch_size: int, seq_len: int | None = None) -> int:
    """FLOPs for one optimizer step (forward + backward) on a full batch."""
    T = seq_len or model.cfg.block_size
    return 3 * batch_size * T * analytic_forward_flops_per_token(model, T)


def measure_train_flops(model: nn.Module, batch_size: int = 1, seq_len: int | None = None) -> int:
    """Count FLOPs of one real forward + backward pass with PyTorch's FlopCounterMode.

    Attention runs on the reference ("math") kernel during the measurement so
    every backend is counted the same way. Gradients are cleared afterwards.
    """
    T = seq_len or model.cfg.block_size
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(model.cfg.vocab_size, (batch_size, T), generator=gen).to(device)
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        _, loss = model(idx, idx)
        loss.backward()
    model.zero_grad(set_to_none=True)
    return counter.get_total_flops()


def summarize(cfg: ModelConfig, batch_size: int = 64) -> dict[str, float]:
    model = build_model(cfg)
    per_step = analytic_train_flops(model, batch_size)
    tokens_per_step = batch_size * cfg.block_size
    return {
        "params": count_params(model),
        "non_embedding_params": count_params(model, non_embedding=True),
        "train_flops_per_token": per_step / tokens_per_step,
        "train_flops_per_step": per_step,
        "approx_6N_per_token": 6 * count_params(model, non_embedding=True),
    }


def main() -> None:
    batch_size, iters = 64, 10_000
    for name in ("thinkpad", "baseline"):
        s = summarize(MODEL_PRESETS[name], batch_size)
        print(f"{name}:")
        print(f"  parameters              {s['params'] / 1e6:8.2f} M")
        print(f"  non-embedding params    {s['non_embedding_params'] / 1e6:8.2f} M")
        print(f"  train FLOPs / token     {s['train_flops_per_token'] / 1e6:8.1f} M")
        print(f"  (6N rule of thumb)      {s['approx_6N_per_token'] / 1e6:8.1f} M")
        print(f"  train FLOPs / step      {s['train_flops_per_step']:.3e}  (batch {batch_size})")
        print(f"  {iters:,} steps           {s['train_flops_per_step'] * iters:.3e}")


if __name__ == "__main__":
    main()
