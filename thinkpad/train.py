"""Training loop.

Usage (from the repo root):

    python -m thinkpad.train --arch thinkpad --out_dir runs/thinkpad
    python -m thinkpad.train --arch baseline --out_dir runs/baseline

    # Give both models the same training compute instead of the same step count:
    python -m thinkpad.train --arch baseline --out_dir runs/baseline-matched --flops_budget 1.495e17

Each run directory gets:

    config.json       model + training config
    metrics.jsonl     one JSON line per log/eval step (loss, lr, tokens, FLOPs)
    checkpoint.pt     latest state, for resuming (rewritten every eval)
    best_model.pt     weights with the lowest validation loss
    results.json      final validation and test loss / perplexity of best_model.pt
    sample.txt        a generated sample from best_model.pt

Re-running the same command resumes from checkpoint.pt. Resuming with a
different model config is refused, so two runs can never share a directory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import fields

import torch

from .config import MODEL_PRESETS, ModelConfig, TrainConfig
from .data import TokenData, prepare_wikitext103, split_path
from .flops import analytic_train_flops
from .model import build_model, count_params, load_state_dict_compat

# ── Setup helpers ──────────────────────────────────────────────────────────


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_ctx(device: str, dtype: str):
    if dtype == "float32":
        return contextlib.nullcontext()
    if dtype != "bfloat16":
        raise ValueError(f"unsupported dtype {dtype!r}")
    return torch.amp.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)


def lr_at(it: int, tc: TrainConfig, max_iters: int) -> float:
    """Linear warmup, then cosine decay to ``min_lr`` at ``max_iters``."""
    if it < tc.warmup_iters:
        return tc.learning_rate * (it + 1) / tc.warmup_iters
    if it >= max_iters:
        return tc.min_lr
    ratio = (it - tc.warmup_iters) / max(1, max_iters - tc.warmup_iters)
    return tc.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (tc.learning_rate - tc.min_lr)


def make_optimizer(model: torch.nn.Module, tc: TrainConfig, device: str) -> torch.optim.AdamW:
    """AdamW; weight decay on matrices only (not embeddings, norms or biases)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and "embedding" not in name else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": tc.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.device(device).type == "cuda"
    return torch.optim.AdamW(
        groups, lr=tc.learning_rate, betas=(tc.beta1, tc.beta2), eps=1e-8, fused=fused
    )


# ── Evaluation ─────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_full(model, data: TokenData, ctx) -> float:
    """Exact mean token loss over a whole split (non-overlapping windows)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in data.sequential_batches():
        with ctx:
            _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens


@torch.no_grad()
def eval_random(model, data: TokenData, iters: int, ctx, generator: torch.Generator) -> float:
    """Mean loss over ``iters`` random batches (used to track train loss)."""
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = data.random_batch(generator)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ── Checkpoints ────────────────────────────────────────────────────────────


def save_checkpoint(path, model, optimizer, it, best_val, mc, tc, data_gen) -> None:
    state = {
        "iter": it,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "best_val_loss": best_val,
        "model_config": mc.to_dict(),
        "train_config": tc.to_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data": data_gen.get_state(),
        },
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint behind


# ── Main loop ──────────────────────────────────────────────────────────────


def train(mc: ModelConfig, tc: TrainConfig) -> dict:
    device = resolve_device(tc.device)
    torch.manual_seed(tc.seed)
    os.makedirs(tc.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tc.out_dir, "checkpoint.pt")
    best_path = os.path.join(tc.out_dir, "best_model.pt")
    metrics_path = os.path.join(tc.out_dir, "metrics.jsonl")

    model = build_model(mc).to(device)
    optimizer = make_optimizer(model, tc, device)
    flops_per_iter = analytic_train_flops(model, tc.batch_size)
    tokens_per_iter = tc.batch_size * mc.block_size

    max_iters = tc.max_iters
    if tc.flops_budget is not None:
        max_iters = int(tc.flops_budget // flops_per_iter)
        print(f"FLOPs budget {tc.flops_budget:.3e} -> {max_iters:,} steps")

    data_gen = torch.Generator().manual_seed(tc.seed)
    eval_gen = torch.Generator()
    start_iter, best_val = 0, float("inf")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ModelConfig.from_dict(ckpt["model_config"])
        if saved != mc:
            raise SystemExit(
                f"{ckpt_path} was trained with a different model config:\n  saved:   {saved}\n"
                f"  current: {mc}\nUse a new --out_dir for a new run."
            )
        load_state_dict_compat(model, ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        data_gen.set_state(ckpt["rng"]["data"])
        start_iter, best_val = ckpt["iter"], ckpt["best_val_loss"]
        print(f"Resumed from step {start_iter:,} (best val {best_val:.4f})")
    else:
        with open(os.path.join(tc.out_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "model": mc.to_dict(),
                    "train": tc.to_dict(),
                    "max_iters": max_iters,
                    "flops_per_iter": flops_per_iter,
                },
                f,
                indent=2,
            )

    train_data = TokenData(split_path(tc.data_dir, "train"), mc.block_size, tc.batch_size, device)
    val_data = TokenData(
        split_path(tc.data_dir, "validation"), mc.block_size, tc.batch_size, device
    )

    print(
        f"{mc.arch}: {count_params(model) / 1e6:.2f}M params "
        f"({count_params(model, non_embedding=True) / 1e6:.2f}M non-embedding) on {device}"
    )
    print(
        f"{flops_per_iter:.3e} FLOPs/step, {tokens_per_iter:,} tokens/step, {max_iters:,} steps "
        f"= {flops_per_iter * max_iters:.3e} FLOPs, "
        f"{tokens_per_iter * max_iters / len(train_data):.2f} epochs of {len(train_data):,} tokens"
    )

    step_model = torch.compile(model) if tc.compile else model
    ctx = autocast_ctx(device, tc.dtype)

    def record(**row) -> None:
        with open(metrics_path, "a") as log:
            log.write(json.dumps(row) + "\n")

    t0 = time.time()
    for it in range(start_iter, max_iters + 1):
        lr = lr_at(it, tc, max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        progress = {"iter": it, "tokens": it * tokens_per_iter, "flops": it * flops_per_iter}

        already_evaluated = it == start_iter and start_iter > 0  # done before the resume
        if (it % tc.eval_interval == 0 or it == max_iters) and not already_evaluated:
            val_loss = eval_full(step_model, val_data, ctx)
            train_loss = eval_random(
                step_model, train_data, tc.eval_iters, ctx, eval_gen.manual_seed(tc.seed + it)
            )
            print(
                f"step {it:>6}: train {train_loss:.4f}  val {val_loss:.4f}  "
                f"ppl {math.exp(val_loss):.1f}  lr {lr:.2e}  ({time.time() - t0:.0f}s)"
            )
            record(kind="eval", **progress, train_loss=train_loss, val_loss=val_loss, lr=lr)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, model, None, it, best_val, mc, tc, data_gen)
            save_checkpoint(ckpt_path, model, optimizer, it, best_val, mc, tc, data_gen)

        if it == max_iters:
            break

        x, y = train_data.random_batch(data_gen)
        with ctx:
            _, loss = step_model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()

        if it % tc.log_interval == 0:
            record(kind="train", **progress, loss=loss.item(), lr=lr)

    return finalize(mc, tc, device, ctx)


def finalize(mc: ModelConfig, tc: TrainConfig, device: str, ctx) -> dict:
    """Evaluate best_model.pt on validation and test, and save a sample."""
    model = build_model(mc).to(device)
    best = torch.load(
        os.path.join(tc.out_dir, "best_model.pt"), map_location=device, weights_only=False
    )
    load_state_dict_compat(model, best["model_state"])

    results = {"arch": mc.arch, "best_iter": best["iter"], "params": count_params(model)}
    for split in ("validation", "test"):
        data = TokenData(split_path(tc.data_dir, split), mc.block_size, tc.batch_size, device)
        loss = eval_full(model, data, ctx)
        results[f"{split}_loss"] = loss
        results[f"{split}_ppl"] = math.exp(loss)
    with open(os.path.join(tc.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    if tc.sample_tokens <= 0:
        return results

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    torch.manual_seed(tc.seed)
    prompt = torch.tensor([enc.encode_ordinary("The history of")], device=device)
    out = model.generate(prompt, tc.sample_tokens, temperature=0.8, top_k=40)
    text = enc.decode(out[0].tolist())
    with open(os.path.join(tc.out_dir, "sample.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    return results


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv=None) -> tuple[ModelConfig, TrainConfig, bool]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--arch",
        default="thinkpad",
        choices=sorted(MODEL_PRESETS),
        help="model preset (default: thinkpad)",
    )
    for f in fields(ModelConfig):
        if f.name != "arch":
            ap.add_argument(
                f"--{f.name}",
                type=type(f.default),
                default=None,
                help=f"override the preset's {f.name}",
            )
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        d = getattr(defaults, f.name)
        if isinstance(d, bool):
            ap.add_argument(f"--{f.name}", action="store_true", help=f"enable {f.name}")
        else:
            ap.add_argument(
                f"--{f.name}",
                type=float if f.name == "flops_budget" else type(d),
                default=d,
                help=f"default: {d}",
            )
    ap.add_argument(
        "--skip_prepare",
        action="store_true",
        help="don't download/tokenize WikiText-103 (data must already exist)",
    )
    args = vars(ap.parse_args(argv))

    preset = MODEL_PRESETS[args.pop("arch")]
    model_over = {f.name: args.pop(f.name) for f in fields(ModelConfig) if f.name != "arch"}
    mc = ModelConfig(
        **{**preset.to_dict(), **{k: v for k, v in model_over.items() if v is not None}}
    )
    skip_prepare = args.pop("skip_prepare")
    return mc, TrainConfig(**args), skip_prepare


def main(argv=None) -> None:
    mc, tc, skip_prepare = parse_args(argv)
    if not skip_prepare:
        prepare_wikitext103(tc.data_dir)
    train(mc, tc)


if __name__ == "__main__":
    main()
