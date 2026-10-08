import numpy as np
import torch

from thinkpad.data import TokenData, write_tokens

EOT = 999


def fake_encode_batch(lines):
    return [[ord(c) for c in line] for line in lines]


def test_write_tokens_skips_blank_lines_and_appends_eot(tmp_path):
    path = str(tmp_path / "t.bin")
    n = write_tokens(["ab\n", "   \n", "", "c\n"], fake_encode_batch, EOT, path, chunk_size=1)
    tokens = np.fromfile(path, dtype=np.uint16).tolist()
    assert tokens == [97, 98, 10, EOT, 99, 10, EOT]
    assert n == len(tokens)
    assert not (tmp_path / "t.bin.tmp").exists()


def make_data(tmp_path, n_tokens=1000, block_size=8, batch_size=4):
    path = str(tmp_path / "d.bin")
    np.arange(n_tokens, dtype=np.uint16).tofile(path)
    return TokenData(path, block_size, batch_size, "cpu")


def test_random_batch_targets_are_shifted_inputs(tmp_path):
    data = make_data(tmp_path)
    x, y = data.random_batch(torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.long
    assert torch.equal(y, x + 1)  # tokens are 0, 1, 2, ... so the next token is +1


def test_random_batch_is_reproducible(tmp_path):
    data = make_data(tmp_path)
    a = data.random_batch(torch.Generator().manual_seed(7))
    b = data.random_batch(torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0])


def test_sequential_batches_cover_split_once(tmp_path):
    data = make_data(tmp_path, n_tokens=100, block_size=8, batch_size=4)
    xs = torch.cat([x for x, _ in data.sequential_batches()])
    assert xs.shape == ((100 - 1) // 8, 8)  # 12 full windows; last batch is partial
    assert torch.equal(xs.flatten(), torch.arange(xs.numel()))  # contiguous, no overlap
