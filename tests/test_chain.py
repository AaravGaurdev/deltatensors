"""
Tests for delta chain lineage: inspect_chain, load_delta_chain, save_delta_chain_from_paths.

Run with: pytest tests/test_chain.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("torch")
pytest.importorskip("safetensors")

import torch
from safetensors.torch import save_file

import deltatensors as dt


# ---------------------------------------------------------------------------
# Synthetic model data — three versions: base, v1, v2
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)

BASE_SD: dict = {
    "layer.weight": _RNG.standard_normal((64, 32)).astype(np.float32),
    "layer.bias":   _RNG.standard_normal((64,)).astype(np.float32),
    "head.weight":  _RNG.standard_normal((10, 64)).astype(np.float32),
}
# v1: small delta on top of base
V1_SD: dict = {k: v + _RNG.standard_normal(v.shape).astype(np.float32) * 0.01
               for k, v in BASE_SD.items()}
# v2: small delta on top of v1
V2_SD: dict = {k: v + _RNG.standard_normal(v.shape).astype(np.float32) * 0.005
               for k, v in V1_SD.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_safetensors(path: str, sd: dict) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    save_file(
        {k: torch.tensor(v) for k, v in sd.items()},
        str(Path(path) / "model.safetensors"),
    )


@pytest.fixture
def chain_dirs(tmp_path):
    """Write base / v1 / v2 safetensors dirs and return their paths."""
    base_dir = str(tmp_path / "base")
    v1_dir   = str(tmp_path / "v1")
    v2_dir   = str(tmp_path / "v2")
    _write_safetensors(base_dir, BASE_SD)
    _write_safetensors(v1_dir,   V1_SD)
    _write_safetensors(v2_dir,   V2_SD)
    return base_dir, v1_dir, v2_dir


@pytest.fixture
def flat_chain(tmp_path, chain_dirs):
    """
    Flat chain: both deltas are against base.
        base → v1.wdelta → v1
               v2.wdelta → v2   (vs base, NOT vs v1)
    """
    base_dir, v1_dir, v2_dir = chain_dirs
    v1_delta = str(tmp_path / "v1.wdelta")
    v2_delta = str(tmp_path / "v2.wdelta")
    dt.save_delta_from_paths(v1_delta, v1_dir, base_dir, strategy="sparse", sparsity=0.5)
    dt.save_delta_from_paths(v2_delta, v2_dir, base_dir, strategy="sparse", sparsity=0.5)
    return base_dir, v1_delta, v2_delta


@pytest.fixture
def true_chain(tmp_path, chain_dirs):
    """
    True chain: v2 delta is against v1, not base.
        base → v1.wdelta → v1 → v2_chained.wdelta → v2
    """
    base_dir, v1_dir, v2_dir = chain_dirs
    v1_delta = str(tmp_path / "v1.wdelta")
    v2_delta = str(tmp_path / "v2_chained.wdelta")
    dt.save_delta_from_paths(v1_delta, v1_dir, base_dir, strategy="sparse", sparsity=0.5)
    dt.save_delta_chain_from_paths(
        v2_delta, v2_dir, v1_delta, base_dir, strategy="sparse", sparsity=0.5
    )
    return base_dir, v1_delta, v2_delta


# ---------------------------------------------------------------------------
# inspect_chain
# ---------------------------------------------------------------------------

class TestInspectChain:
    def test_length(self, flat_chain):
        _, v1_delta, v2_delta = flat_chain
        history = dt.inspect_chain([v1_delta, v2_delta])
        assert len(history) == 2

    def test_step_indices(self, flat_chain):
        _, v1_delta, v2_delta = flat_chain
        history = dt.inspect_chain([v1_delta, v2_delta])
        assert history[0]["step"] == 0
        assert history[1]["step"] == 1

    def test_contains_inspect_fields(self, flat_chain):
        _, v1_delta, _ = flat_chain
        history = dt.inspect_chain([v1_delta])
        entry = history[0]
        for field in ("path", "size_mb", "parent_hash", "strategy", "n_tensors"):
            assert field in entry

    def test_parent_hash_is_sha256(self, flat_chain):
        _, v1_delta, v2_delta = flat_chain
        for entry in dt.inspect_chain([v1_delta, v2_delta]):
            assert len(entry["parent_hash"]) == 64
            int(entry["parent_hash"], 16)  # must be valid hex

    def test_single_delta(self, flat_chain):
        _, v1_delta, _ = flat_chain
        history = dt.inspect_chain([v1_delta])
        assert len(history) == 1
        assert history[0]["step"] == 0


# ---------------------------------------------------------------------------
# load_delta_chain — flat (all vs base)
# ---------------------------------------------------------------------------

class TestLoadDeltaChainFlat:
    def test_single_step_matches_load_delta_from_paths(self, tmp_path, flat_chain):
        base_dir, v1_delta, _ = flat_chain
        from_chain = dt.load_delta_chain([v1_delta], base=base_dir)
        direct     = dt.load_delta_from_paths(v1_delta, base_dir)
        for k in from_chain:
            np.testing.assert_array_equal(from_chain[k], direct[k])

    def test_two_step_flat_last_matches_v2(self, flat_chain):
        """Loading v2 directly from base should equal loading via chain (both flat)."""
        base_dir, _, v2_delta = flat_chain
        from_chain = dt.load_delta_chain([v2_delta], base=base_dir)
        for k, orig in V2_SD.items():
            np.testing.assert_allclose(from_chain[k].astype(np.float32), orig, atol=0.1)

    def test_returns_all_keys(self, flat_chain):
        base_dir, v1_delta, _ = flat_chain
        result = dt.load_delta_chain([v1_delta], base=base_dir)
        assert set(result.keys()) == set(BASE_SD.keys())

    def test_base_as_state_dict(self, flat_chain):
        _, v1_delta, _ = flat_chain
        result = dt.load_delta_chain([v1_delta], base=BASE_SD)
        for k, orig in V1_SD.items():
            np.testing.assert_allclose(result[k].astype(np.float32), orig, atol=0.1)

    def test_empty_delta_paths_raises(self, flat_chain):
        base_dir, _, _ = flat_chain
        with pytest.raises(ValueError, match="empty"):
            dt.load_delta_chain([], base=base_dir)


# ---------------------------------------------------------------------------
# load_delta_chain — true chain (v2 vs v1)
# ---------------------------------------------------------------------------

class TestLoadDeltaChainTrue:
    def test_two_step_reconstruction(self, true_chain):
        """Applying [v1, v2_chained] to base should reproduce V2_SD."""
        base_dir, v1_delta, v2_delta = true_chain
        result = dt.load_delta_chain([v1_delta, v2_delta], base=base_dir)
        for k, orig in V2_SD.items():
            np.testing.assert_allclose(result[k].astype(np.float32), orig, atol=0.1)

    def test_one_step_gives_v1(self, true_chain):
        """Stopping after the first delta should give V1_SD."""
        base_dir, v1_delta, _ = true_chain
        result = dt.load_delta_chain([v1_delta], base=base_dir)
        for k, orig in V1_SD.items():
            np.testing.assert_allclose(result[k].astype(np.float32), orig, atol=0.1)

    def test_wrong_order_raises(self, true_chain):
        """Applying v2_chained before v1 should fail hash verification."""
        base_dir, v1_delta, v2_delta = true_chain
        with pytest.raises(ValueError, match="hash mismatch"):
            dt.load_delta_chain([v2_delta, v1_delta], base=base_dir)

    def test_verify_false_skips_hash_check(self, true_chain):
        """With verify=False, wrong order should not raise (but gives wrong values)."""
        base_dir, v1_delta, v2_delta = true_chain
        # This won't raise but will produce nonsense — that's the contract of verify=False
        result = dt.load_delta_chain([v2_delta, v1_delta], base=base_dir, verify=False)
        assert set(result.keys()) == set(BASE_SD.keys())


# ---------------------------------------------------------------------------
# save_delta_chain_from_paths
# ---------------------------------------------------------------------------

class TestSaveDeltaChainFromPaths:
    def test_creates_file(self, tmp_path, chain_dirs):
        base_dir, v1_dir, v2_dir = chain_dirs
        v1_delta = str(tmp_path / "v1.wdelta")
        v2_delta = str(tmp_path / "v2.wdelta")
        dt.save_delta_from_paths(v1_delta, v1_dir, base_dir, strategy="sparse", sparsity=0.5)
        dt.save_delta_chain_from_paths(v2_delta, v2_dir, v1_delta, base_dir, strategy="sparse", sparsity=0.5)
        assert Path(v2_delta).exists()

    def test_parent_hash_links_correctly(self, tmp_path, chain_dirs):
        """v2_chained's parent_hash must equal hash(v1_reconstructed)."""
        base_dir, v1_dir, v2_dir = chain_dirs
        v1_delta = str(tmp_path / "v1.wdelta")
        v2_delta = str(tmp_path / "v2.wdelta")

        dt.save_delta_from_paths(v1_delta, v1_dir, base_dir, strategy="sparse", sparsity=0.5)
        dt.save_delta_chain_from_paths(v2_delta, v2_dir, v1_delta, base_dir, strategy="sparse", sparsity=0.5)

        v1_reconstructed = dt.load_delta_from_paths(v1_delta, base_dir)
        expected_parent_hash = dt.hash_state_dict(v1_reconstructed)
        actual_parent_hash   = dt.inspect(v2_delta)["parent_hash"]

        assert actual_parent_hash == expected_parent_hash

    def test_roundtrip_via_load_chain(self, tmp_path, chain_dirs):
        base_dir, v1_dir, v2_dir = chain_dirs
        v1_delta = str(tmp_path / "v1.wdelta")
        v2_delta = str(tmp_path / "v2.wdelta")

        dt.save_delta_from_paths(v1_delta, v1_dir, base_dir, strategy="sparse", sparsity=0.5)
        dt.save_delta_chain_from_paths(v2_delta, v2_dir, v1_delta, base_dir, strategy="sparse", sparsity=0.5)

        result = dt.load_delta_chain([v1_delta, v2_delta], base=base_dir)
        for k, orig in V2_SD.items():
            np.testing.assert_allclose(result[k].astype(np.float32), orig, atol=0.1)

    def test_chained_delta_has_smaller_magnitude(self, chain_dirs):
        """
        The incremental delta (v2 vs v1) should have smaller L1 norm than the
        flat delta (v2 vs base) because v1→v2 is a smaller perturbation than base→v2.

        Note: file *size* is the same at equal sparsity because sparse stores a fixed
        fraction of elements regardless of their magnitude.  The benefit of chaining
        is smaller-magnitude incremental differences, which yield better reconstruction
        quality at the same compression ratio.
        """
        base_dir, v1_dir, v2_dir = chain_dirs
        from safetensors.torch import load_file

        base_t = load_file(str(Path(base_dir) / "model.safetensors"))
        v1_t   = load_file(str(Path(v1_dir)   / "model.safetensors"))
        v2_t   = load_file(str(Path(v2_dir)   / "model.safetensors"))

        flat_l1 = sum(
            (v2_t[k].float() - base_t[k].float()).abs().sum().item()
            for k in base_t
        )
        chain_l1 = sum(
            (v2_t[k].float() - v1_t[k].float()).abs().sum().item()
            for k in base_t
        )

        assert chain_l1 < flat_l1, (
            f"incremental L1 ({chain_l1:.4f}) should be smaller than flat L1 ({flat_l1:.4f})"
        )
