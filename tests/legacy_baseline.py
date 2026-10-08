"""The original Colab baseline GPT architecture, kept verbatim as a test reference.

Used only by tests/test_equivalence.py to check that ``thinkpad.model`` computes
exactly the same function. The original code reads hyperparameters from module
globals, so ``build`` sets them before constructing the model.
"""
# ruff: noqa
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

n_embd = n_head = n_layer = block_size = vocab_size = None
dropout = 0.0
device = "cpu"


def build(cfg):
    globals().update(n_embd=cfg.n_embd, n_head=cfg.n_head, n_layer=cfg.n_layer,
                     block_size=cfg.block_size, vocab_size=cfg.vocab_size, dropout=cfg.dropout)
    model = BaselineGPT()
    mark_residual_projections(model)
    model.apply(model._init_weights)
    return model


# ── Original code below (unchanged) ─────────────────────────────────────

# OPTIMIZATION: Vectorized MultiHeadAttention replacing Head + ModuleList loop
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.n_head = num_heads
        self.head_size = head_size
        # Single matrix for all Q, K, V
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj   = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.size()

        # Calculate Q, K, V for all heads in parallel
        q, k, v = self.c_attn(x).split(n_embd, dim=2)

        # Reshape for Flash Attention: (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Flash Attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        # Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BaselineGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RESIDUAL_PROJ'):
                std = 0.02 / math.sqrt(2 * n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def mark_residual_projections(model):
    for block in model.blocks:
        block.sa.proj.RESIDUAL_PROJ    = True
        block.ffwd.net[2].RESIDUAL_PROJ = True

