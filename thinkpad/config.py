"""Model and training configuration.

The two presets reproduce the comparison this repo was built for: a dual-stream
Think-Pad model and a vanilla GPT baseline sized to a similar parameter count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

GPT2_VOCAB_SIZE = 50257


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "thinkpad"  # "thinkpad" or "baseline"
    vocab_size: int = GPT2_VOCAB_SIZE
    block_size: int = 512  # context length in tokens
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.arch not in ("thinkpad", "baseline"):
            raise ValueError(f"unknown arch {self.arch!r}; expected 'thinkpad' or 'baseline'")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

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
}


@dataclass(frozen=True)
class TrainConfig:
    out_dir: str = "runs/thinkpad"
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
    dtype: str = "bfloat16"  # autocast dtype: "bfloat16" or "float32"
    compile: bool = False
    # If set, overrides max_iters so the run spends this many training FLOPs.
    flops_budget: float | None = None
    sample_tokens: int = 200  # length of the text sample written at the end (0 to skip)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **kwargs: Any) -> TrainConfig:
        return replace(self, **kwargs)
