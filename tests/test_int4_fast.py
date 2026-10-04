"""
The LUT-based int4 decompression must match the 0.2.0 reference implementation
on v1 payloads, and v2 payloads must agree with v1 to within one quantization step.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltatensors.compress_int4 import compress_int4, decompress_int4, decompress_int4_add_  # noqa: E402
from deltatensors.format import write_wdelta, read_wdelta  # noqa: E402
from int4_reference import _compress_int4_v1_reference, _decompress_int4_reference  # noqa: E402

SHAPES = [(1, 2), (3,), (7, 5), (64, 33), (128, 256), (3_000_001,)]


def _delta(shape, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 0.01).astype(np.float32)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("frac", [0.0, 0.01, 0.2])
def test_v1_matches_reference(shape, frac):
    payload = _compress_int4_v1_reference(_delta(shape), outlier_fraction=frac)
    ref = _decompress_int4_reference(payload)
    got = decompress_int4(payload)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert np.max(np.abs(got - ref)) <= 1e-6
    np.testing.assert_array_equal(got, ref)  # in fact bit-exact


@pytest.mark.parametrize("shape", SHAPES)
def test_v1_add_inplace_matches_reference(shape):
    payload = _compress_int4_v1_reference(_delta(shape, 1))
    base = np.random.default_rng(2).standard_normal(shape).astype(np.float32)
    want = base + _decompress_int4_reference(payload)
    out = base.copy()
    assert decompress_int4_add_(payload, out) is out
    np.testing.assert_array_equal(out, want)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("frac", [0.0, 0.01, 0.2])
def test_v2_within_one_step_of_v1(shape, frac):
    delta = _delta(shape, 3)
    p1 = _compress_int4_v1_reference(delta, outlier_fraction=frac)
    p2 = compress_int4(delta, outlier_fraction=frac)
    assert p2["int4_layout"] == 2 and p2["outlier_idx"].dtype == np.int32
    assert len(p2["packed"]) == (delta.size + 1) // 2
    np.testing.assert_array_equal(p1["outlier_idx"], p2["outlier_idx"])
    np.testing.assert_array_equal(p1["outlier_vals"], p2["outlier_vals"])

    v1 = _decompress_int4_reference(p1)
    v2 = decompress_int4(p2)
    step = float(p1["scale"][0])
    assert np.max(np.abs(v1 - v2)) <= step * 1.001
    # and each is within half a step (+ f16 rounding) of the true delta off-outlier
    err = np.abs(v2 - delta).reshape(-1)
    err[p2["outlier_idx"]] = 0
    assert err.max() <= step * 0.51 + 1e-3 * np.abs(delta).max()


@pytest.mark.parametrize("shape", SHAPES)
def test_v2_add_inplace(shape):
    delta = _delta(shape, 4)
    p2 = compress_int4(delta)
    base = np.random.default_rng(5).standard_normal(shape).astype(np.float32)
    out = base.copy()
    decompress_int4_add_(p2, out)
    np.testing.assert_array_equal(out, base + decompress_int4(p2))
    idx = p2["outlier_idx"]
    flat = out.reshape(-1)
    np.testing.assert_array_equal(
        flat[idx], base.reshape(-1)[idx] + p2["outlier_vals"].astype(np.float32)
    )


def test_single_element_tensor():
    # 0.2.0 crashed here (every element is an outlier, empty quantization range)
    for frac in (0.0, 0.5):
        p = compress_int4(np.array([0.5], np.float32), outlier_fraction=frac)
        np.testing.assert_array_equal(decompress_int4(p), np.array([0.5], np.float32))


def test_constant_delta():
    p1 = _compress_int4_v1_reference(np.full((10, 10), 0.25, np.float32))
    np.testing.assert_array_equal(decompress_int4(p1), _decompress_int4_reference(p1))
    p2 = compress_int4(np.full((10, 10), 0.25, np.float32))
    np.testing.assert_allclose(decompress_int4(p2), 0.25, atol=1e-3)


def test_v2_roundtrip_through_file():
    delta = _delta((33, 17), 6)
    buf = io.BytesIO()
    write_wdelta(buf, "ab" * 32, "int4", {"w": compress_int4(delta)})
    buf.seek(0)
    _, _, tensors = read_wdelta(buf)
    p = tensors["w"]
    assert p["int4_layout"] == 2
    np.testing.assert_array_equal(decompress_int4(p), decompress_int4(compress_int4(delta)))


def test_add_rejects_bad_out():
    payload = compress_int4(_delta((4, 4)))
    with pytest.raises(ValueError):
        decompress_int4_add_(payload, np.zeros(16, np.float16))
    with pytest.raises(ValueError):
        decompress_int4_add_(payload, np.zeros(15, np.float32))


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 64])
@pytest.mark.parametrize("frac", [0.0, 0.05, 0.5, 0.99])
def test_tiny_chunks_cross_boundaries(monkeypatch, chunk, frac):
    import deltatensors.compress_int4 as m
    monkeypatch.setattr(m, "_CHUNK", chunk)
    for shape in [(1, 2), (7,), (9, 13), (257,)]:
        delta = _delta(shape, 7)
        p1 = _compress_int4_v1_reference(delta, outlier_fraction=frac)
        np.testing.assert_array_equal(decompress_int4(p1), _decompress_int4_reference(p1))
        p2 = compress_int4(delta, outlier_fraction=frac)
        base = np.ones(shape, np.float32)
        out = base.copy()
        decompress_int4_add_(p2, out)
        monkeypatch.setattr(m, "_CHUNK", 1 << 20)
        np.testing.assert_array_equal(out, base + decompress_int4(p2))
        monkeypatch.setattr(m, "_CHUNK", chunk)
