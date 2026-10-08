"""The original Colab Think-Pad architecture, kept verbatim as a test reference.

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
    model = ThinkPadGPT()
    mark_residual_projections(model)
    model.apply(model._init_weights)
    return model


# ── Original code below (unchanged) ─────────────────────────────────────

# OPTIMIZATION: Vectorized MultiHeadAttention for both Self and Cross Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.n_head = num_heads
        self.head_size = head_size
        # Separate matrices to cleanly handle cross-attention without shape math errors
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key   = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x, kv_src=None):
        B, T, C = x.size()
        if kv_src is None:
            kv_src = x  # Self-attention fallback

        # Calculate Q, K, V
        q = self.query(x)
        k = self.key(kv_src)
        v = self.value(kv_src)

        # Reshape for Flash Attention: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Flash Attention handles the causal mask correctly since T matches for both streams
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        # Reassemble
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd, input_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = n_embd
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GatedCrossConnect(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.gate_proj = nn.Linear(2 * n_embd, n_embd, bias=False)

    def forward(self, tgt, src):
        gate = torch.sigmoid(self.gate_proj(torch.cat([tgt, src], dim=-1)))
        return gate * src + (1 - gate) * tgt


class ThinkPadBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        # Step 1
        self.sa_x     = MultiHeadAttention(n_head, head_size)
        self.ln_x1    = nn.LayerNorm(n_embd)

        # Step 2
        self.mha_p1   = MultiHeadAttention(n_head, head_size)
        self.ln_p1_p  = nn.LayerNorm(n_embd)
        self.ln_p1_x  = nn.LayerNorm(n_embd)

        # Step 3
        self.mha_p2   = MultiHeadAttention(n_head, head_size)
        self.ln_p2_p  = nn.LayerNorm(n_embd)
        self.ln_p2_x  = nn.LayerNorm(n_embd)

        # Step 4
        self.gate_x   = GatedCrossConnect(n_embd)
        self.gate_p   = GatedCrossConnect(n_embd)
        self.ln_x_post = nn.LayerNorm(n_embd)
        self.ln_p_post = nn.LayerNorm(n_embd)

        # Step 5
        self.mha_bypass = MultiHeadAttention(n_head, head_size)
        self.ln_by_x    = nn.LayerNorm(n_embd)
        self.ln_by_p    = nn.LayerNorm(n_embd)

        # Step 6
        self.ffwd_x   = FeedForward(n_embd, input_dim=2 * n_embd)
        self.ln_x_ffn = nn.LayerNorm(n_embd)

        # Step 7
        self.ffwd_p   = FeedForward(n_embd)
        self.ln_p_ffn = nn.LayerNorm(n_embd)

    def forward(self, x, p):
        x = x + self.sa_x(self.ln_x1(x))
        p = p + self.mha_p1(self.ln_p1_p(p), kv_src=self.ln_p1_x(x))
        p = p + self.mha_p2(self.ln_p2_p(p), kv_src=self.ln_p2_x(x))

        x_pre = x
        x_gated = self.gate_x(x, p)
        x = self.ln_x_post(x + x_gated)

        p_gated = self.gate_p(p, x_pre)
        p = self.ln_p_post(p + p_gated)

        bypass = self.mha_bypass(self.ln_by_x(x), kv_src=self.ln_by_p(p))

        x_input = torch.cat([self.ln_x_ffn(x), bypass], dim=-1)
        x = x + self.ffwd_x(x_input)

        p = p + self.ffwd_p(self.ln_p_ffn(p))

        return x, p


class ThinkPadGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.ModuleList([ThinkPadBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.ln_p_f  = nn.LayerNorm(n_embd)
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
        p = torch.zeros_like(x)

        for block in self.blocks:
            x, p = block(x, p)

        x = self.ln_f(x)
        p = self.ln_p_f(p)

        logits = self.lm_head(p)

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
        for mha in [block.sa_x, block.mha_p1, block.mha_p2, block.mha_bypass]:
            mha.proj.RESIDUAL_PROJ = True
        block.ffwd_x.net[2].RESIDUAL_PROJ = True
        block.ffwd_p.net[2].RESIDUAL_PROJ = True

