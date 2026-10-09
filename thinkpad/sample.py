"""Generate text from a checkpoint.

    python -m thinkpad.sample runs/thinkpad/best_model.pt --prompt "The history of"

Checkpoints from the original Colab scripts work too; they don't record which
architecture they are, so pass ``--arch thinkpad`` or ``--arch baseline``.
"""

from __future__ import annotations

import argparse

import torch

from .config import ModelConfig
from .data import load_codec
from .model import _LanguageModel, build_model, load_state_dict_compat


def load_model(path: str, arch: str | None = None, device: str = "cpu") -> _LanguageModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_config" in ckpt:
        cfg = ModelConfig.from_dict(ckpt["model_config"])
    elif "config" in ckpt:  # original Colab checkpoint
        if arch is None:
            raise SystemExit("this is an original Colab checkpoint; pass --arch thinkpad|baseline")
        cfg = ModelConfig.from_dict({**ckpt["config"], "arch": arch})
    else:
        raise SystemExit(f"{path} doesn't look like a Think-Pad checkpoint")
    model = build_model(cfg).to(device)
    dropped = load_state_dict_compat(model, ckpt["model_state"])
    if dropped:
        print(f"(ignored {len(dropped)} unused tensors from the original checkpoint format)")
    return model.eval()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint")
    ap.add_argument("--prompt", default="The history of")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--arch", choices=["thinkpad", "baseline"], default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--data_dir",
        default=None,
        help="dataset dir (for the tokenizer); default: the one the run trained on",
    )
    args = ap.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_dir = args.data_dir or ckpt.get("train_config", {}).get("data_dir", "data/wikitext103")
    encode, decode = load_codec(data_dir)
    model = load_model(args.checkpoint, args.arch, args.device)
    torch.manual_seed(args.seed)
    prompt = torch.tensor([encode(args.prompt)], device=args.device)
    for i in range(args.num_samples):
        out = model.generate(prompt, args.max_new_tokens, args.temperature, args.top_k)
        print(f"--- sample {i + 1} ---\n{decode(out[0].tolist())}\n")


if __name__ == "__main__":
    main()
