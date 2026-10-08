"""Plot validation loss against training compute for one or more runs.

    python -m thinkpad.plot runs/thinkpad runs/baseline --out assets/val_loss_vs_flops.png

Comparing at equal FLOPs, not equal steps, is what makes the comparison fair:
the two architectures do different amounts of work per step.
"""

from __future__ import annotations

import argparse
import json
import os


def read_evals(run_dir: str) -> list[dict]:
    """Eval rows from metrics.jsonl; if a resumed run logged a step twice, keep the last."""
    rows: dict[int, dict] = {}
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") == "eval":
                rows[row["iter"]] = row
    return [rows[k] for k in sorted(rows)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="run directories containing metrics.jsonl")
    ap.add_argument("--labels", nargs="*", help="legend labels (default: directory names)")
    ap.add_argument("--out", default="assets/val_loss_vs_flops.png")
    ap.add_argument(
        "--skip_first",
        type=int,
        default=1,
        help="leave out the first N evals (step 0 loss is ~10.8 and squashes the plot)",
    )
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    if len(labels) != len(args.runs):
        ap.error("give one --labels entry per run")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for run, label in zip(args.runs, labels, strict=True):
        rows = read_evals(run)[args.skip_first :]
        ax.plot(
            [r["flops"] for r in rows],
            [r["val_loss"] for r in rows],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=label,
        )
    ax.set_xlabel("Training compute (FLOPs)")
    ax.set_ylabel("Validation loss (nats/token)")
    ax.set_title("WikiText-103 validation loss vs. training compute")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
