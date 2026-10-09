"""Model and training configuration.

The two presets reproduce the comparison this repo was built for: a dual-stream
Think-Pad model and a vanilla GPT baseline sized to a similar parameter count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

GPT2_VOCAB_SIZE = 50257

# Think-Pad block steps that can be switched off, per layer, for ablations.
# Steps 1 (x self-attention) and 6 (x FFN) are the GPT backbone and always run;
# step 6 can still be ablated with ``ffn_x`` for completeness.
STEP_FIELDS = {
    "p_read1": "step 2: p attends to x",
    "p_read2": "step 3: p attends to x again",
    "gate_x": "step 4: gate p into x (+ post-LN on x)",
    "gate_p": "step 4: gate x into p (+ post-LN on p)",
    "bypass": "step 5: x attends to p, result feeds x's FFN",
    "ffn_x": "step 6: x FFN",
    "ffn_p": "step 7: p FFN",
}


def layer_set(spec: str, n_layer: int) -> frozenset[int]:
    """Which layers a step runs in.

    ``all`` | ``none`` | ``even`` | ``odd`` | ``first:K`` | ``last:K`` (K may be
    ``half``) | comma-separated indices such as ``0,3,5``.
    """
    spec = spec.strip().lower()
    if spec == "all":
        return frozenset(range(n_layer))
    if spec == "none":
        return frozenset()
    if spec == "even":
        return frozenset(range(0, n_layer, 2))
    if spec == "odd":
        return frozenset(range(1, n_layer, 2))
    if spec.startswith(("first:", "last:")):
        where, k = spec.split(":", 1)
        k = n_layer // 2 if k == "half" else int(k)
        if not 0 <= k <= n_layer:
            raise ValueError(f"layer spec {spec!r} out of range for {n_layer} layers")
        return frozenset(range(k)) if where == "first" else frozenset(range(n_layer - k, n_layer))
    try:
        idx = frozenset(int(i) for i in spec.split(",") if i.strip())
    except ValueError:
        raise ValueError(f"bad layer spec {spec!r}") from None
    if any(not 0 <= i < n_layer for i in idx):
        raise ValueError(f"layer spec {spec!r} out of range for {n_layer} layers")
    return idx


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "thinkpad"  # "thinkpad" or "baseline"
    vocab_size: int = GPT2_VOCAB_SIZE
    block_size: int = 512  # context length in tokens
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 8
    dropout: float = 0.1
    # Think-Pad step switches (layer specs, see ``layer_set``). Defaults = full model.
    p_read1: str = "all"
    p_read2: str = "all"
    gate_x: str = "all"
    gate_p: str = "all"
    bypass: str = "all"
    ffn_x: str = "all"
    ffn_p: str = "all"

    def __post_init__(self) -> None:
        if self.arch not in ("thinkpad", "baseline"):
            raise ValueError(f"unknown arch {self.arch!r}; expected 'thinkpad' or 'baseline'")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        steps = {name: self.layers(name) for name in STEP_FIELDS}  # validates every spec
        if self.arch == "thinkpad" and not (steps["p_read1"] | steps["p_read2"] | steps["gate_p"]):
            raise ValueError("p never reads from x: enable p_read1, p_read2 or gate_p somewhere")

    def layers(self, step: str) -> frozenset[int]:
        return layer_set(getattr(self, step), self.n_layer)

    def ablated_steps(self) -> dict[str, str]:
        """Step switches that differ from the full model."""
        return {k: getattr(self, k) for k in STEP_FIELDS if getattr(self, k) != "all"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


MODEL_PRESETS: dict[str, ModelConfig] = {
    # Dual-stream model used in the experiments.
    "thinkpad": ModelConfig(arch="thinkpad", n_embd=384, n_head=6, n_layer=8),
    # Vanilla GPT, 13 layers so its parameter count is close to Think-Pad's.
    "baseline": ModelConfig(arch="baseline", n_embd=512, n_head=8, n_layer=13),
    # Small versions for CPU smoke tests and debugging.
    "thinkpad-tiny": ModelConfig(arch="thinkpad", block_size=64, n_embd=64, n_head=4, n_layer=2),
    "baseline-tiny": ModelConfig(arch="baseline", block_size=64, n_embd=64, n_head=4, n_layer=3),
    # Character-level pair for quick CPU pilots on Tiny Shakespeare (65-symbol vocabulary).
    # 12 GPT layers roughly match 4 Think-Pad layers in parameters.
    "thinkpad-char": ModelConfig(
        arch="thinkpad", vocab_size=65, block_size=64, n_embd=128, n_head=4, n_layer=4
    ),
    "baseline-char": ModelConfig(
        arch="baseline", vocab_size=65, block_size=64, n_embd=128, n_head=4, n_layer=12
    ),
}


@dataclass(frozen=True)
class TrainConfig:
    out_dir: str = "runs/thinkpad"
    dataset: str = "wikitext103"  # "wikitext103" or "shakespeare_char"
    data_dir: str = "data/wikitext103"
    batch_size: int = 64
    max_iters: int = 10_000
    eval_interval: int = 500
    eval_iters: int = 100  # random batches used to estimate train loss at each eval
    log_interval: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 1_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "auto"  # "auto", "cuda", "cpu" or "mps"
    # "auto": bfloat16 on GPUs that support it (A100, H100, L4, RTX 30xx+), else float32.
    dtype: str = "auto"  # "auto", "bfloat16" or "float32"
    compile: bool = False
    # If set, overrides max_iters so the run spends this many training FLOPs.
    flops_budget: float | None = None
    sample_tokens: int = 200  # length of the text sample written at the end (0 to skip)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **kwargs: Any) -> TrainConfig:
        return replace(self, **kwargs)
