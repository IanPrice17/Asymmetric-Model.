"""WikiText-103 tokenization and batching.

Each split is stored once as a flat ``uint16`` array of GPT-2 token ids
(``train.bin``, ``validation.bin``, ``test.bin``) and memory-mapped for training.
Every non-empty raw line is tokenized as-is and followed by ``<|endoftext|>``,
matching the preprocessing used for the original runs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator

import numpy as np
import torch

SPLITS = ("train", "validation", "test")


def split_path(data_dir: str, split: str) -> str:
    return os.path.join(data_dir, f"{split}.bin")


def write_tokens(
    lines: Iterable[str],
    encode_batch: Callable[[list[str]], list[list[int]]],
    eot_token: int,
    path: str,
    chunk_size: int = 10_000,
) -> int:
    """Tokenize ``lines`` in chunks and append them to ``path``. Returns the token count."""
    tmp_path = path + ".tmp"
    total = 0
    with open(tmp_path, "wb") as f:
        chunk: list[str] = []

        def flush() -> int:
            ids: list[int] = []
            for toks in encode_batch(chunk):
                ids.extend(toks)
                ids.append(eot_token)
            np.asarray(ids, dtype=np.uint16).tofile(f)
            chunk.clear()
            return len(ids)

        for line in lines:
            if line.strip():
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    total += flush()
        if chunk:
            total += flush()
    os.replace(tmp_path, path)  # only a complete file ever has the final name
    return total


def prepare_wikitext103(data_dir: str) -> None:
    """Download WikiText-103 (raw) and tokenize every split that is not on disk yet."""
    missing = [s for s in SPLITS if not os.path.exists(split_path(data_dir, s))]
    if not missing:
        return
    import tiktoken
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    print("Loading WikiText-103 (raw, ~500 MB on first download)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    for split in missing:
        n = write_tokens(
            dataset[split]["text"],
            lambda batch: enc.encode_ordinary_batch(batch, num_threads=8),
            enc.eot_token,
            split_path(data_dir, split),
        )
        print(f"  {split}: {n:,} tokens -> {split_path(data_dir, split)}")


class TokenData:
    """A memory-mapped token file that serves (input, target) batches."""

    def __init__(self, path: str, block_size: int, batch_size: int, device: str | torch.device):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run data preparation first")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) <= block_size:
            raise ValueError(f"{path} has {len(self.data)} tokens, fewer than block_size + 1")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.data)

    def _to_device(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.device.type == "cuda":
            x, y = x.pin_memory(), y.pin_memory()
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def _stack(self, starts: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.block_size
        x = torch.stack([torch.from_numpy(self.data[i : i + T].astype(np.int64)) for i in starts])
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + T].astype(np.int64)) for i in starts]
        )
        return self._to_device(x, y)

    def random_batch(self, generator: torch.Generator | None = None):
        """``batch_size`` windows at uniformly random offsets (with replacement)."""
        ix = torch.randint(
            len(self.data) - self.block_size, (self.batch_size,), generator=generator
        )
        return self._stack(ix.tolist())

    def sequential_batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Non-overlapping windows covering the whole file once, for exact evaluation.

        Each window starts with an empty context, so the resulting loss is
        slightly higher than a sliding-window evaluation would give.
        """
        starts = list(range(0, len(self.data) - self.block_size, self.block_size))
        for i in range(0, len(starts), self.batch_size):
            yield self._stack(starts[i : i + self.batch_size])
