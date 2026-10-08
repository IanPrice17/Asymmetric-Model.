import pytest

from thinkpad import MODEL_PRESETS, build_model
from thinkpad.flops import analytic_train_flops, measure_train_flops


@pytest.mark.parametrize("preset", ["thinkpad-tiny", "baseline-tiny"])
@pytest.mark.parametrize("seq_len", [None, 16])
def test_analytic_matches_pytorch_count(preset, seq_len):
    model = build_model(MODEL_PRESETS[preset])
    assert analytic_train_flops(model, 2, seq_len) == measure_train_flops(model, 2, seq_len)


def test_flops_scale_linearly_with_batch():
    model = build_model(MODEL_PRESETS["thinkpad-tiny"])
    assert measure_train_flops(model, 3) == 3 * measure_train_flops(model, 1)


def test_measurement_leaves_no_gradients():
    model = build_model(MODEL_PRESETS["baseline-tiny"])
    measure_train_flops(model, 1)
    assert all(p.grad is None for p in model.parameters())
