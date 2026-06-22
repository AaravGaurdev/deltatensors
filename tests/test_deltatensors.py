"""
Tests for deltatensors.

Run with: pytest tests/
"""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import deltatensors as dt
from deltatensors.compress import compress, decompress
from deltatensors.lineage import hash_state_dict, verify_base
from deltatensors.format import write_wdelta, read_wdelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_state_dict(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return {
        "layer1.weight": rng.standard_normal((64, 32)).astype(np.float32) * scale,
        "layer1.bias":   rng.standard_normal((64,)).astype(np.float32) * scale,
        "layer2.weight": rng.standard_normal((32, 64)).astype(np.float32) * scale,
        "layer2.bias":   rng.standard_normal((32,)).astype(np.float32) * scale,
    }


BASE = make_state_dict(seed=0)
FINETUNED = {k: v + make_state_dict(seed=1)[k] * 0.01 for k, v in BASE.items()}


# ---------------------------------------------------------------------------
# Compress / decompress round-trip
# ---------------------------------------------------------------------------

class TestSparseRoundTrip:
    def test_exact_at_zero_sparsity(self):
        delta = np.random.randn(64, 32).astype(np.float32)
        payload = compress(delta, "sparse", sparsity=0.0)
        recovered = decompress(payload)
        np.testing.assert_allclose(delta, recovered, rtol=1e-5)

    def test_shape_preserved(self):
        delta = np.random.randn(8, 16, 4).astype(np.float32)
        payload = compress(delta, "sparse", sparsity=0.5)
        recovered = decompress(payload)
        assert recovered.shape == delta.shape

    def test_sparsity_reduces_values_stored(self):
        delta = np.random.randn(1000).astype(np.float32)
        p90 = compress(delta, "sparse", sparsity=0.9)
        p50 = compress(delta, "sparse", sparsity=0.5)
        assert len(p90["indices"]) < len(p50["indices"])

    def test_invalid_sparsity_raises(self):
        delta = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            compress(delta, "sparse", sparsity=1.0)


class TestQuantizedRoundTrip:
    def test_shape_preserved(self):
        delta = np.random.randn(32, 64).astype(np.float32)
        payload = compress(delta, "quantized")
        recovered = decompress(payload)
        assert recovered.shape == delta.shape

    def test_sign_agreement(self):
        """Reconstructed signs should match original signs on >95% of elements."""
        delta = np.random.randn(256, 256).astype(np.float32)
        payload = compress(delta, "quantized")
        recovered = decompress(payload)
        sign_match = np.sign(delta.flatten()) == np.sign(recovered.flatten())
        assert sign_match.mean() > 0.95

    def test_1d_delta(self):
        delta = np.random.randn(64).astype(np.float32)
        payload = compress(delta, "quantized")
        recovered = decompress(payload)
        assert recovered.shape == delta.shape


class TestUnknownStrategy:
    def test_compress_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            compress(np.ones((4,), dtype=np.float32), "svd_magic")


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

class TestLineage:
    def test_same_dict_same_hash(self):
        assert hash_state_dict(BASE) == hash_state_dict(BASE)

    def test_different_dict_different_hash(self):
        other = make_state_dict(seed=99)
        assert hash_state_dict(BASE) != hash_state_dict(other)

    def test_verify_passes_correct_base(self):
        h = hash_state_dict(BASE)
        verify_base(BASE, h)  # should not raise

    def test_verify_raises_wrong_base(self):
        wrong = make_state_dict(seed=42)
        h = hash_state_dict(BASE)
        with pytest.raises(ValueError, match="hash mismatch"):
            verify_base(wrong, h)

    def test_key_order_invariant(self):
        sd1 = {"a": np.ones((2,), dtype=np.float32), "b": np.zeros((2,), dtype=np.float32)}
        sd2 = {"b": np.zeros((2,), dtype=np.float32), "a": np.ones((2,), dtype=np.float32)}
        assert hash_state_dict(sd1) == hash_state_dict(sd2)


# ---------------------------------------------------------------------------
# Format (binary serialisation)
# ---------------------------------------------------------------------------

class TestFormat:
    def _make_compressed(self, strategy):
        tensors = {}
        for name in BASE:
            delta = (FINETUNED[name] - BASE[name]).astype(np.float32)
            tensors[name] = compress(delta, strategy)
        return tensors

    def test_sparse_roundtrip(self):
        tensors = self._make_compressed("sparse")
        buf = io.BytesIO()
        write_wdelta(buf, "deadbeef", "sparse", tensors)
        buf.seek(0)
        ph, strat, recovered = read_wdelta(buf)
        assert ph == "deadbeef"
        assert strat == "sparse"
        assert set(recovered.keys()) == set(tensors.keys())

    def test_quantized_roundtrip(self):
        tensors = self._make_compressed("quantized")
        buf = io.BytesIO()
        write_wdelta(buf, "cafebabe", "quantized", tensors)
        buf.seek(0)
        ph, strat, recovered = read_wdelta(buf)
        assert strat == "quantized"

    def test_bad_magic_raises(self):
        buf = io.BytesIO(b"XXXX\x01\x00\x00\x00\x00\x00\x00\x00")
        with pytest.raises(ValueError):  # drop the match=
            read_wdelta(buf)


# -------------------------python--------------------------------------------------
# End-to-end: save_delta / load_delta
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_sparse_reconstruction_close(self, tmp_path):
        path = tmp_path / "model.wdelta"
        dt.save_delta(path, FINETUNED, BASE, strategy="sparse", sparsity=0.8)
        recon = dt.load_delta(path, BASE)
        for name in BASE:
            corr = np.corrcoef(recon[name].flatten(), FINETUNED[name].flatten())[0, 1]
            assert corr > 0.99, f"Low correlation for '{name}' at sparsity=0.8: {corr:.4f}"

    def test_sparse_lossless_at_zero_sparsity(self, tmp_path):
        path = tmp_path / "model.wdelta"
        dt.save_delta(path, FINETUNED, BASE, strategy="sparse", sparsity=0.0)
        recon = dt.load_delta(path, BASE)
        for name in BASE:
            np.testing.assert_allclose(recon[name], FINETUNED[name], rtol=1e-5, atol=1e-6)

    def test_quantized_reconstruction_reasonable(self, tmp_path):
        path = tmp_path / "model.wdelta"
        dt.save_delta(path, FINETUNED, BASE, strategy="quantized")
        recon = dt.load_delta(path, BASE)
        # Quantized is lossy: check correlation, not exact values
        for name in BASE:
            ft_flat = FINETUNED[name].flatten()
            re_flat = recon[name].flatten()
            corr = np.corrcoef(ft_flat, re_flat)[0, 1]
            assert corr > 0.9, f"Low correlation for {name}: {corr:.3f}"

    def test_wrong_base_raises(self, tmp_path):
        path = tmp_path / "model.wdelta"
        dt.save_delta(path, FINETUNED, BASE, strategy="sparse")
        wrong_base = make_state_dict(seed=99)
        with pytest.raises(ValueError, match="hash mismatch"):
            dt.load_delta(path, wrong_base)

    def test_file_size_smaller_than_finetuned(self, tmp_path):
        path = tmp_path / "model.wdelta"
        dt.save_delta(path, FINETUNED, BASE, strategy="sparse", sparsity=0.9)
        ft_size = sum(v.nbytes for v in FINETUNED.values())
        wdelta_size = path.stat().st_size
        assert wdelta_size < ft_size, f".wdelta ({wdelta_size}B) not smaller than finetuned ({ft_size}B)"

    def test_inspect(self, tmp_path):
        path = tmp_path / "model.wdelta"
        dt.save_delta(path, FINETUNED, BASE, strategy="sparse")
        info = dt.inspect(path)
        assert info["strategy"] == "sparse"
        assert info["n_tensors"] == len(BASE)
        assert "parent_hash" in info

    def test_key_mismatch_raises(self, tmp_path):
        path = tmp_path / "model.wdelta"
        bad_ft = {k: v for k, v in FINETUNED.items() if k != "layer1.bias"}
        with pytest.raises(ValueError, match="Key mismatch"):
            dt.save_delta(path, bad_ft, BASE, strategy="sparse")

    def test_shape_mismatch_raises(self, tmp_path):
        path = tmp_path / "model.wdelta"
        bad_ft = dict(FINETUNED)
        bad_ft["layer1.weight"] = np.random.randn(128, 32).astype(np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            dt.save_delta(path, bad_ft, BASE, strategy="sparse")
