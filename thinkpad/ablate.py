"""Ablation sweep: which Think-Pad steps earn their cost?

Every variant is trained on the same *training compute* (FLOPs) as the full
model, so a variant that drops a step gets proportionally more optimizer steps.
That makes the question "is this step worth its compute?", not just "does this
step help?".

Phases (run in this order; pick with --phases):

  cut       p reads x once; then also drop the p->x bypass into x's FFN
  isolate   remove one step at a time from the full model
  layers    run the cross-stream steps (2-5) only in some layers; self-attention
            and both FFNs still run in every layer
  combine   the cuts that helped, together
  baseline  plain GPT at the same compute, for reference

Usage:

    # quick CPU pilot on Tiny Shakespeare (small character-level models)
    python -m thinkpad.ablate --model thinkpad-char --baseline baseline-char \\
        --dataset shakespeare_char --data_dir data/shakespeare_char --steps 600 \\
        --out_root runs/ablate-char -- --batch_size 32 --eval_interval 100 --eval_iters 20

    # full-size on a GPU (WikiText-103)
    python -m thinkpad.ablate --out_root runs/ablate --steps 2500

Arguments after ``--`` go to ``thinkpad.train`` for every run. Re-running the
same command skips finished variants and resumes interrupted ones. Results go
to ``<out_root>/summary.md``, ``summary.json`` and ``val_loss_vs_flops.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass, field

from .config import MODEL_PRESETS
from .flops import analytic_train_flops
from .model import build_model, count_params

CROSS_STEPS = ("p_read1", "p_read2", "gate_x", "gate_p", "bypass")


@dataclass(frozen=True)
class Variant:
    name: str
    phase: str
    what: str
    steps: dict[str, str] = field(default_factory=dict)
    baseline: bool = False


VARIANTS = [
    Variant("full", "cut", "all 7 steps (reference)"),
    Variant("one_read", "cut", "p reads x once (no step 3)", {"p_read2": "none"}),
    Variant(
        "one_read_no_bypass",
        "cut",
        "one read, and no x->p bypass into x's FFN (no steps 3, 5)",
        {"p_read2": "none", "bypass": "none"},
    ),
    Variant("no_gate_x", "isolate", "no gate from p into x (step 4, x side)", {"gate_x": "none"}),
    Variant("no_gate_p", "isolate", "no gate from x into p (step 4, p side)", {"gate_p": "none"}),
    Variant("no_bypass", "isolate", "no x->p bypass into x's FFN (step 5)", {"bypass": "none"}),
    Variant("no_ffn_p", "isolate", "no p FFN (step 7)", {"ffn_p": "none"}),
    Variant(
        "cross_even",
        "layers",
        "steps 2-5 only in even layers (0, 2, ...)",
        dict.fromkeys(CROSS_STEPS, "even"),
    ),
    Variant(
        "cross_first_half",
        "layers",
        "steps 2-5 only in the first half of layers",
        dict.fromkeys(CROSS_STEPS, "first:half"),
    ),
    Variant(
        "cross_last_half",
        "layers",
        "steps 2-5 only in the second half of layers",
        dict.fromkeys(CROSS_STEPS, "last:half"),
    ),
    Variant(
        "lean",
        "combine",
        "one read, no bypass, no x->p gate (no steps 3, 5, 4p)",
        {"p_read2": "none", "bypass": "none", "gate_p": "none"},
    ),
    Variant(
        "lean_first_half",
        "combine",
        "lean, with the remaining cross steps (2, 4x) only in the first half",
        {
            "p_read2": "none",
            "bypass": "none",
            "gate_p": "none",
            "p_read1": "first:half",
            "gate_x": "first:half",
        },
    ),
    Variant("gpt_baseline", "baseline", "plain GPT, same compute", baseline=True),
]
PHASES = ("cut", "isolate", "layers", "combine", "baseline")


def run_dir(out_root: str, v: Variant, seed: int, n_seeds: int) -> str:
    return os.path.join(out_root, v.name if n_seeds == 1 else f"{v.name}-seed{seed}")


def run_variant(v: Variant, args, seed: int, budget: float, passthrough: list[str]) -> None:
    from .train import main as train_main

    out = run_dir(args.out_root, v, seed, args.seeds)
    if os.path.exists(os.path.join(out, "results.json")):
        print(f"[{v.name} seed {seed}] done already, skipping")
        return
    print(f"\n=== {v.name} (seed {seed}): {v.what} ===")
    argv = ["--arch", args.baseline if v.baseline else args.model]
    for step, spec in v.steps.items():
        argv += [f"--{step}", spec]
    argv += [
        "--out_dir", out, "--data_dir", args.data_dir, "--dataset", args.dataset,
        "--flops_budget", repr(budget), "--seed", str(seed), "--skip_prepare",
    ]  # fmt: skip
    train_main(argv + passthrough)


def summarize(args, variants: list[Variant], seeds: list[int]) -> list[dict]:
    from .plot import read_evals

    rows = []
    for v in variants:
        runs = []
        for seed in seeds:
            out = run_dir(args.out_root, v, seed, args.seeds)
            path = os.path.join(out, "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    res = json.load(f)
                with open(os.path.join(out, "config.json")) as f:
                    cfg = json.load(f)
                runs.append((out, res, cfg))
        if not runs:
            continue
        vals = [r["validation_loss"] for _, r, _ in runs]
        tests = [r["test_loss"] for _, r, _ in runs]
        _, res, cfg = runs[0]
        rows.append(
            {
                "variant": v.name,
                "phase": v.phase,
                "what": v.what,
                "params_M": res["params"] / 1e6,
                "non_embedding_params_M": res["non_embedding_params"] / 1e6,
                "flops_per_step": cfg["flops_per_iter"],
                "steps": cfg["max_iters"],
                "val_loss": statistics.mean(vals),
                "val_loss_sd": statistics.stdev(vals) if len(vals) > 1 else None,
                "test_loss": statistics.mean(tests),
                "seeds": len(runs),
                "curve_dirs": [o for o, _, _ in runs],
            }
        )
    full = next((r for r in rows if r["variant"] == "full"), None)
    for r in rows:
        r["delta_val_vs_full"] = r["val_loss"] - full["val_loss"] if full else None

    with open(os.path.join(args.out_root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    header = ["Rank", "Variant", "What changed", "Params (M)", "FLOPs/step", "Steps",
              "Val loss", "Δ vs full", "Test loss"]  # fmt: skip
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for i, r in enumerate(sorted(rows, key=lambda r: r["val_loss"]), 1):
        sd = f" ± {r['val_loss_sd']:.3f}" if r["val_loss_sd"] is not None else ""
        delta = "–" if r["delta_val_vs_full"] is None else f"{r['delta_val_vs_full']:+.3f}"
        lines.append(
            f"| {i} | {r['variant']} | {r['what']} | {r['params_M']:.2f} | "
            f"{r['flops_per_step']:.2e} | {r['steps']:,} | {r['val_loss']:.3f}{sd} | {delta} | "
            f"{r['test_loss']:.3f} |"
        )
    table = "\n".join(lines)
    seeds_note = (
        f"{len(seeds)} seed(s); ± is the standard deviation across seeds."
        if len(seeds) > 1
        else "1 seed: differences under ~0.01 are within run-to-run noise."
    )
    with open(os.path.join(args.out_root, "summary.md"), "w") as f:
        f.write(
            f"# Think-Pad ablations\n\nModel `{args.model}`, data `{args.dataset}`, "
            f"every run at the full model's compute for {args.steps:,} steps. "
            f"Loss in nats/token (lower is better). {seeds_note}\n\n{table}\n"
        )
    print("\n" + table)

    try:  # plot
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.set_prop_cycle(color=plt.get_cmap("tab20").colors)  # >10 distinct colors
        for r in sorted(rows, key=lambda r: r["val_loss"]):
            ev = read_evals(r["curve_dirs"][0])[1:]
            style = "--" if r["phase"] == "baseline" else "-"
            lw = 2.4 if r["variant"] == "full" else 1.4
            ax.plot([e["flops"] for e in ev], [e["val_loss"] for e in ev], style, lw=lw,
                    label=f"{r['variant']} ({r['val_loss']:.3f})")  # fmt: skip
        ax.set_xlabel("Training compute (FLOPs)")
        ax.set_ylabel("Validation loss (nats/token)")
        ax.set_title("Think-Pad ablations at equal training compute")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_root, "val_loss_vs_flops.png"))
    except ImportError:
        pass
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="thinkpad", choices=sorted(MODEL_PRESETS))
    ap.add_argument("--baseline", default="baseline", choices=sorted(MODEL_PRESETS))
    ap.add_argument("--dataset", default="wikitext103")
    ap.add_argument("--data_dir", default="data/wikitext103")
    ap.add_argument("--out_root", default="runs/ablate")
    ap.add_argument("--steps", type=int, default=2500,
                    help="steps for the full model; sets every variant's FLOPs budget")  # fmt: skip
    ap.add_argument("--batch_size", type=int, default=None,
                    help="batch size for every run (default: training default)")  # fmt: skip
    ap.add_argument("--phases", nargs="+", default=list(PHASES), choices=PHASES)
    ap.add_argument("--only", nargs="+", default=None, help="run just these variant names")
    ap.add_argument("--seeds", type=int, default=1, help="repeat each variant with N seeds")
    ap.add_argument("--summary_only", action="store_true")
    argv = list(argv) if argv is not None else None
    import sys

    raw = sys.argv[1:] if argv is None else argv
    passthrough = raw[raw.index("--") + 1 :] if "--" in raw else []
    args = ap.parse_args(raw[: raw.index("--")] if "--" in raw else raw)
    if args.batch_size is not None:
        passthrough = ["--batch_size", str(args.batch_size), *passthrough]
    batch_size = args.batch_size
    for i, a in enumerate(passthrough[:-1]):
        if a == "--batch_size":
            batch_size = int(passthrough[i + 1])
    if batch_size is None:
        from .config import TrainConfig

        batch_size = TrainConfig().batch_size

    variants = [v for v in VARIANTS if v.phase in args.phases]
    if args.only:
        variants = [v for v in variants if v.name in args.only]
    seeds = [1337 + i for i in range(args.seeds)]
    os.makedirs(args.out_root, exist_ok=True)

    full = build_model(MODEL_PRESETS[args.model])
    budget = analytic_train_flops(full, batch_size) * args.steps
    print(f"Budget: {budget:.3e} FLOPs per run (= {args.model}, {args.steps:,} steps, "
          f"batch {batch_size}); full model {count_params(full) / 1e6:.2f}M params")  # fmt: skip

    if not args.summary_only:
        from .data import prepare

        prepare(args.dataset, args.data_dir)
        for seed in seeds:  # one full pass over the variants per seed, so results come early
            for v in variants:
                run_variant(v, args, seed, math.nextafter(budget, math.inf), passthrough)
    summarize(args, variants, seeds)


if __name__ == "__main__":
    main()
