"""Think-Pad: a dual-stream transformer, plus the vanilla GPT baseline it is compared with.

Think-Pad keeps two residual streams of the same width:

* ``x``, the backbone, which looks like an ordinary GPT stream;
* ``p``, the "pad", which starts at zero, reads from ``x`` and is the stream
  the next-token prediction is made from.

Each block (see ``ThinkPadBlock.forward``):

1. ``x`` self-attention (pre-norm residual).
2. ``p`` attends to ``x`` (pre-norm residual).
3. ``p`` attends to ``x`` again with separate weights.
4. Gated exchange: each stream mixes in the other through a learned sigmoid
   gate, followed by a post-LayerNorm on both streams.
5. ``x`` attends to ``p``; the result is not added to ``x`` but is fed to x's FFN.
6. ``x`` FFN on ``[LN(x), bypass]`` (residual).
7. ``p`` FFN (residual).

All attention is causal. Cross-stream attention is causal too, because both
streams index the same token positions: position ``t`` of one stream only
reads positions ``<= t`` of the other.

Module and parameter names match the original Colab scripts, so checkpoints
trained with them load here (see ``load_state_dict_compat``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig

# ── Shared layers ─────────────────────────────────────────────────────────


class MultiHeadAttention(nn.Module):
    """Causal multi-head attention with optional cross-stream keys and values.

    Queries come from ``x``; keys and values come from ``kv_src`` (``x`` itself
    when omitted). Separate Q/K/V projections keep cross-attention simple.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.key = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.RESIDUAL_PROJ = True
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, kv_src: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        if kv_src is None:
            kv_src = x
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(kv_src).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(kv_src).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class FusedSelfAttention(nn.Module):
    """Causal self-attention with a fused QKV projection (used by the baseline)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_size = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.RESIDUAL_PROJ = True
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig, input_dim: int | None = None):
        super().__init__()
        n = cfg.n_embd
        self.net = nn.Sequential(
            nn.Linear(input_dim or n, 4 * n, bias=False),
            nn.GELU(),
            nn.Linear(4 * n, n, bias=False),
            nn.Dropout(cfg.dropout),
        )
        self.net[2].RESIDUAL_PROJ = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedCrossConnect(nn.Module):
    """``g * src + (1 - g) * tgt`` with ``g = sigmoid(W [tgt; src])``, per channel."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(2 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, tgt: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(torch.cat([tgt, src], dim=-1)))
        return gate * src + (1 - gate) * tgt


# ── Base class: embeddings, init, loss, generation ──────────────────────────


class _LanguageModel(nn.Module):
    cfg: ModelConfig

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_PROJ", False):
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        if self.cfg.block_size < T:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        return self.token_embedding_table(idx) + self.position_embedding_table(pos)

    def _logits_and_loss(self, h: torch.Tensor, targets: torch.Tensor | None):
        logits = self.lm_head(h)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train(was_training)
        return idx


# ── Think-Pad ──────────────────────────────────────────────────────────────


class ThinkPadBlock(nn.Module):
    """One dual-stream block.

    In the final block, steps 4–6 for ``x`` are skipped: the loss is computed
    from ``p`` only, and after step 4 nothing flows from ``x`` back into ``p``,
    so those layers would never receive a gradient. Leaving them out does not
    change the function the model computes.
    """

    def __init__(self, cfg: ModelConfig, is_last: bool = False):
        super().__init__()
        n = cfg.n_embd
        self.is_last = is_last

        # Step 1: x self-attention
        self.sa_x = MultiHeadAttention(cfg)
        self.ln_x1 = nn.LayerNorm(n)
        # Step 2: p reads x
        self.mha_p1 = MultiHeadAttention(cfg)
        self.ln_p1_p = nn.LayerNorm(n)
        self.ln_p1_x = nn.LayerNorm(n)
        # Step 3: p reads x again (separate weights)
        self.mha_p2 = MultiHeadAttention(cfg)
        self.ln_p2_p = nn.LayerNorm(n)
        self.ln_p2_x = nn.LayerNorm(n)
        # Step 4: gated exchange between streams, then post-LN
        self.gate_p = GatedCrossConnect(cfg)
        self.ln_p_post = nn.LayerNorm(n)
        # Step 7: p FFN
        self.ffwd_p = FeedForward(cfg)
        self.ln_p_ffn = nn.LayerNorm(n)

        if not is_last:
            # Step 4 (x side)
            self.gate_x = GatedCrossConnect(cfg)
            self.ln_x_post = nn.LayerNorm(n)
            # Step 5: x reads p; result feeds x's FFN only
            self.mha_bypass = MultiHeadAttention(cfg)
            self.ln_by_x = nn.LayerNorm(n)
            self.ln_by_p = nn.LayerNorm(n)
            # Step 6: x FFN on [LN(x), bypass]
            self.ffwd_x = FeedForward(cfg, input_dim=2 * n)
            self.ln_x_ffn = nn.LayerNorm(n)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.sa_x(self.ln_x1(x))
        p = p + self.mha_p1(self.ln_p1_p(p), kv_src=self.ln_p1_x(x))
        p = p + self.mha_p2(self.ln_p2_p(p), kv_src=self.ln_p2_x(x))

        # Step 4: both gates read the streams as they were *before* the exchange.
        x_pre, p_pre = x, p
        if not self.is_last:
            x = self.ln_x_post(x_pre + self.gate_x(x_pre, p_pre))
        p = self.ln_p_post(p_pre + self.gate_p(p_pre, x_pre))

        if not self.is_last:
            # Steps 5–6: x reads the updated p; the result only enters x's FFN.
            bypass = self.mha_bypass(self.ln_by_x(x), kv_src=self.ln_by_p(p))
            x = x + self.ffwd_x(torch.cat([self.ln_x_ffn(x), bypass], dim=-1))

        # Step 7
        p = p + self.ffwd_p(self.ln_p_ffn(p))
        return x, p


class ThinkPadGPT(_LanguageModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            [ThinkPadBlock(cfg, is_last=(i == cfg.n_layer - 1)) for i in range(cfg.n_layer)]
        )
        self.ln_p_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight  # weight tying
        self.apply(self._init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self._embed(idx)
        p = torch.zeros_like(x)
        for block in self.blocks:
            x, p = block(x, p)
        return self._logits_and_loss(self.ln_p_f(p), targets)


# ── Baseline GPT ───────────────────────────────────────────────────────────


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.sa = FusedSelfAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class BaselineGPT(_LanguageModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight  # weight tying
        self.apply(self._init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self._embed(idx)
        for block in self.blocks:
            x = block(x)
        return self._logits_and_loss(self.ln_f(x), targets)


# ── Helpers ────────────────────────────────────────────────────────────────


def build_model(cfg: ModelConfig) -> _LanguageModel:
    return ThinkPadGPT(cfg) if cfg.arch == "thinkpad" else BaselineGPT(cfg)


def count_params(model: nn.Module, non_embedding: bool = False) -> int:
    """Unique parameters (tied weights counted once).

    With ``non_embedding=True``, token and position embeddings are excluded,
    the convention used in scaling-law papers.
    """
    n = sum(p.numel() for p in model.parameters())
    if non_embedding:
        n -= model.token_embedding_table.weight.numel()
        n -= model.position_embedding_table.weight.numel()
    return n


def load_state_dict_compat(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Load a checkpoint, including ones saved by the original Colab scripts.

    Old Think-Pad checkpoints contain the never-trained final-block ``x`` layers
    and an unused ``ln_f``; those keys are dropped. Any *missing* key is still an
    error. Returns the list of dropped keys.
    """
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    expected = model.state_dict().keys()
    dropped = sorted(k for k in state_dict if k not in expected)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in expected}, strict=True)
    return dropped
