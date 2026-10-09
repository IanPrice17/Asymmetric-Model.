"""Think-Pad and the GPT baseline.

Two residual streams, same width:
  x - the backbone, basically a normal GPT stream (gets the token embeddings)
  p - the "pad", starts at zero, reads from x, and is what we predict from

Block steps (any of 2-7 can be switched off per layer, see config.py):
  1. x self-attention
  2. p attends to x
  3. p attends to x again (separate weights)
  4. gated exchange between x and p, then LayerNorm on each
  5. x attends to p, result goes into x's FFN (not the residual)
  6. x FFN
  7. p FFN

Cross-attention between the streams is still causal since both streams share
token positions. Param names match my original Colab notebooks so old
checkpoints still load.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import STEP_FIELDS, ModelConfig


class MultiHeadAttention(nn.Module):
    # q comes from x, k/v from kv_src (or x if not given). separate projections
    # so the same module works for self- and cross-attention
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

    def forward(self, x, kv_src=None):
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
    # standard GPT attention w/ one qkv matrix, only used by the baseline
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_size = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.RESIDUAL_PROJ = True
        self.dropout = cfg.dropout

    def forward(self, x):
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
    def __init__(self, cfg: ModelConfig, input_dim=None):
        super().__init__()
        n = cfg.n_embd
        self.net = nn.Sequential(
            nn.Linear(input_dim or n, 4 * n, bias=False),
            nn.GELU(),
            nn.Linear(4 * n, n, bias=False),
            nn.Dropout(cfg.dropout),
        )
        self.net[2].RESIDUAL_PROJ = True

    def forward(self, x):
        return self.net(x)


class GatedCrossConnect(nn.Module):
    # g = sigmoid(W [tgt; src]) per channel, returns g*src + (1-g)*tgt
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(2 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, tgt, src):
        gate = torch.sigmoid(self.gate_proj(torch.cat([tgt, src], dim=-1)))
        return gate * src + (1 - gate) * tgt


class _LanguageModel(nn.Module):
    # shared bits: init, embeddings, loss, sampling
    cfg: ModelConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_PROJ", False):
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer)  # GPT-2 style scaled init
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, idx):
        T = idx.shape[1]
        if self.cfg.block_size < T:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        return self.token_embedding_table(idx) + self.position_embedding_table(pos)

    def _logits_and_loss(self, h, targets):
        logits = self.lm_head(h)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
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


class ThinkPadBlock(nn.Module):
    # Only builds the steps that are switched on for this layer. Since the loss
    # only uses p, any x work after the last layer where p reads x can never get
    # a gradient, so it's skipped too (by default that's steps 4-6 for x in the
    # last block - same function, fewer dead params).

    def __init__(self, cfg: ModelConfig, layer: int, last_x_read: int):
        super().__init__()
        n = cfg.n_embd
        on = {step: layer in cfg.layers(step) for step in STEP_FIELDS}
        x_live = layer < last_x_read  # does x still matter after this layer?

        self.use_sa_x = layer <= last_x_read
        self.use_read1 = on["p_read1"]
        self.use_read2 = on["p_read2"]
        self.use_gate_p = on["gate_p"]
        self.use_ffn_p = on["ffn_p"]
        self.use_sa_p = on["sa_p"]
        self.use_gate_x = on["gate_x"] and x_live
        self.use_ffn_x = on["ffn_x"] and x_live
        self.use_bypass = on["bypass"] and self.use_ffn_x  # bypass only feeds x's FFN

        if self.use_sa_x:  # 1
            self.sa_x = MultiHeadAttention(cfg)
            self.ln_x1 = nn.LayerNorm(n)
        if self.use_read1:  # 2
            self.mha_p1 = MultiHeadAttention(cfg)
            self.ln_p1_p = nn.LayerNorm(n)
            self.ln_p1_x = nn.LayerNorm(n)
        if self.use_read2:  # 3
            self.mha_p2 = MultiHeadAttention(cfg)
            self.ln_p2_p = nn.LayerNorm(n)
            self.ln_p2_x = nn.LayerNorm(n)
        if self.use_sa_p:  # optional p self-attention (not in the original design)
            self.sa_p = MultiHeadAttention(cfg)
            self.ln_p_sa = nn.LayerNorm(n)
        if self.use_gate_x:  # 4, p -> x
            self.gate_x = GatedCrossConnect(cfg)
            self.ln_x_post = nn.LayerNorm(n)
        if self.use_gate_p:  # 4, x -> p
            self.gate_p = GatedCrossConnect(cfg)
            self.ln_p_post = nn.LayerNorm(n)
        if self.use_bypass:  # 5
            self.mha_bypass = MultiHeadAttention(cfg)
            self.ln_by_x = nn.LayerNorm(n)
            self.ln_by_p = nn.LayerNorm(n)
        if self.use_ffn_x:  # 6, input is [LN(x), bypass] when the bypass is on
            self.ffwd_x = FeedForward(cfg, input_dim=2 * n if self.use_bypass else n)
            self.ln_x_ffn = nn.LayerNorm(n)
        if self.use_ffn_p:  # 7
            self.ffwd_p = FeedForward(cfg)
            self.ln_p_ffn = nn.LayerNorm(n)

    def forward(self, x, p):
        if self.use_sa_x:
            x = x + self.sa_x(self.ln_x1(x))
        if self.use_read1:
            p = p + self.mha_p1(self.ln_p1_p(p), kv_src=self.ln_p1_x(x))
        if self.use_read2:
            p = p + self.mha_p2(self.ln_p2_p(p), kv_src=self.ln_p2_x(x))
        if self.use_sa_p:
            p = p + self.sa_p(self.ln_p_sa(p))

        # both gates use the streams from before the swap
        x_pre, p_pre = x, p
        if self.use_gate_x:
            x = self.ln_x_post(x_pre + self.gate_x(x_pre, p_pre))
        if self.use_gate_p:
            p = self.ln_p_post(p_pre + self.gate_p(p_pre, x_pre))

        if self.use_ffn_x:
            h = self.ln_x_ffn(x)
            if self.use_bypass:
                bypass = self.mha_bypass(self.ln_by_x(x), kv_src=self.ln_by_p(p))
                h = torch.cat([h, bypass], dim=-1)
            x = x + self.ffwd_x(h)

        if self.use_ffn_p:
            p = p + self.ffwd_p(self.ln_p_ffn(p))
        return x, p


def last_x_read(cfg: ModelConfig) -> int:
    # last layer where p still pulls anything out of x
    return max(cfg.layers("p_read1") | cfg.layers("p_read2") | cfg.layers("gate_p"))


class ThinkPadGPT(_LanguageModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            [ThinkPadBlock(cfg, i, last_x_read(cfg)) for i in range(cfg.n_layer)]
        )
        self.ln_p_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight  # tied
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        x = self._embed(idx)
        p = torch.zeros_like(x)
        for block in self.blocks:
            x, p = block(x, p)
        return self._logits_and_loss(self.ln_p_f(p), targets)  # predict from p


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.sa = FusedSelfAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
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
        self.lm_head.weight = self.token_embedding_table.weight  # tied
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        x = self._embed(idx)
        for block in self.blocks:
            x = block(x)
        return self._logits_and_loss(self.ln_f(x), targets)


def build_model(cfg: ModelConfig):
    return ThinkPadGPT(cfg) if cfg.arch == "thinkpad" else BaselineGPT(cfg)


def count_params(model, non_embedding=False):
    # tied weights only count once. non_embedding drops token + position
    # embeddings (what scaling-law papers usually report)
    n = sum(p.numel() for p in model.parameters())
    if non_embedding:
        n -= model.token_embedding_table.weight.numel()
        n -= model.position_embedding_table.weight.numel()
    return n


def load_state_dict_compat(model, state_dict):
    # Loads new checkpoints and the old Colab ones. Old Think-Pad checkpoints have
    # the dead last-block x layers + an unused ln_f, so extra keys get dropped.
    # Missing keys still raise. Returns the dropped keys.
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    expected = model.state_dict().keys()
    dropped = sorted(k for k in state_dict if k not in expected)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in expected}, strict=True)
    return dropped
