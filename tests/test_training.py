"""
Tests for DeltaTensorsCallback.

Requires torch + safetensors (skipped automatically if missing).
Run with: pytest tests/test_training.py -v
"""

import os
import shutil
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
from deltatensors.training import DeltaTensorsCallback


# ---------------------------------------------------------------------------
# Synthetic model data (module-level so all tests share the same base hash)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

BASE_SD: dict[str, np.ndarray] = {
    "layer.weight": _RNG.standard_normal((64, 32)).astype(np.float32),
    "layer.bias":   _RNG.standard_normal((64,)).astype(np.float32),
}
FT_SD: dict[str, np.ndarray] = {
    k: v + _RNG.standard_normal(v.shape).astype(np.float32) * 0.01
    for k, v in BASE_SD.items()
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_safetensors(path: str, sd: dict) -> None:
    """Write a state dict as a single model.safetensors file under path."""
    os.makedirs(path, exist_ok=True)
    save_file(
        {k: torch.tensor(v) for k, v in sd.items()},
        os.path.join(path, "model.safetensors"),
    )


def _make_checkpoint(output_dir: str, step: int, source_dir: str) -> str:
    """Copy safetensors from source_dir into output_dir/checkpoint-{step}/."""
    ckpt = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt, exist_ok=True)
    for f in os.listdir(source_dir):
        if f.endswith(".safetensors"):
            shutil.copy(os.path.join(source_dir, f), os.path.join(ckpt, f))
    return ckpt


class _Args:
    def __init__(self, output_dir: str): self.output_dir = output_dir


class _State:
    def __init__(self, step: int): self.global_step = step


class _Control: pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dirs(tmp_path):
    """Write base and finetuned model dirs to a temp location."""
    base_dir = str(tmp_path / "base")
    ft_dir   = str(tmp_path / "finetuned")
    _write_safetensors(base_dir, BASE_SD)
    _write_safetensors(ft_dir, FT_SD)
    return base_dir, ft_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeltaTensorsCallback:
    def test_creates_wdelta(self, tmp_path, dirs):
        base_dir, ft_dir = dirs
        out = str(tmp_path / "outputs")
        _make_checkpoint(out, 100, ft_dir)

        DeltaTensorsCallback(base_dir=base_dir, strategy="int4", outlier_fraction=0.05).on_save(
            _Args(out), _State(100), _Control()
        )

        assert os.path.exists(os.path.join(out, "checkpoint-100", "model.wdelta"))

    def test_roundtrip_int4(self, tmp_path, dirs):
        """Reconstructed weights should be close to original after int4 compression."""
        base_dir, ft_dir = dirs
        out = str(tmp_path / "outputs")
        _make_checkpoint(out, 200, ft_dir)

        DeltaTensorsCallback(base_dir=base_dir, strategy="int4", outlier_fraction=0.05).on_save(
            _Args(out), _State(200), _Control()
        )

        delta_path = os.path.join(out, "checkpoint-200", "model.wdelta")
        recon = dt.load_delta_from_paths(delta_path, base_dir, verify=True)

        assert set(recon.keys()) == set(FT_SD.keys())
        for name, orig in FT_SD.items():
            np.testing.assert_allclose(recon[name].astype(np.float32), orig, atol=0.02)

    def test_roundtrip_sparse(self, tmp_path, dirs):
        """Sparse strategy: inspect confirms correct strategy and tensor count."""
        base_dir, ft_dir = dirs
        out = str(tmp_path / "outputs")
        _make_checkpoint(out, 300, ft_dir)

        DeltaTensorsCallback(base_dir=base_dir, strategy="sparse", sparsity=0.5).on_save(
            _Args(out), _State(300), _Control()
        )

        info = dt.inspect(os.path.join(out, "checkpoint-300", "model.wdelta"))
        assert info["strategy"] == "sparse"
        assert info["n_tensors"] == len(FT_SD)

    def test_delete_full_checkpoint(self, tmp_path, dirs):
        """delete_full_checkpoint=True removes .safetensors after saving delta."""
        base_dir, ft_dir = dirs
        out = str(tmp_path / "outputs")
        ckpt = _make_checkpoint(out, 400, ft_dir)
        assert any(f.endswith(".safetensors") for f in os.listdir(ckpt))

        DeltaTensorsCallback(
            base_dir=base_dir, strategy="sparse", sparsity=0.9,
            delete_full_checkpoint=True,
        ).on_save(_Args(out), _State(400), _Control())

        assert not any(f.endswith(".safetensors") for f in os.listdir(ckpt))
        assert os.path.exists(os.path.join(ckpt, "model.wdelta"))

    def test_safetensors_preserved_by_default(self, tmp_path, dirs):
        """delete_full_checkpoint=False (default) keeps the original safetensors."""
        base_dir, ft_dir = dirs
        out = str(tmp_path / "outputs")
        ckpt = _make_checkpoint(out, 450, ft_dir)
        sf_before = [f for f in os.listdir(ckpt) if f.endswith(".safetensors")]

        DeltaTensorsCallback(base_dir=base_dir, strategy="sparse").on_save(
            _Args(out), _State(450), _Control()
        )

        sf_after = [f for f in os.listdir(ckpt) if f.endswith(".safetensors")]
        assert sf_before == sf_after

    def test_missing_checkpoint_dir_warns(self, tmp_path, dirs, capsys):
        """Graceful warning when checkpoint dir does not exist."""
        base_dir, _ = dirs
        DeltaTensorsCallback(base_dir=base_dir).on_save(
            _Args(str(tmp_path / "outputs")), _State(999), _Control()
        )
        assert "Warning" in capsys.readouterr().out

    def test_missing_safetensors_warns(self, tmp_path, dirs, capsys):
        """Graceful warning when checkpoint dir has no .safetensors files."""
        base_dir, _ = dirs
        ckpt = os.path.join(str(tmp_path / "outputs"), "checkpoint-500")
        os.makedirs(ckpt)
        open(os.path.join(ckpt, "optimizer.pt"), "w").close()  # non-safetensors file

        DeltaTensorsCallback(base_dir=base_dir).on_save(
            _Args(str(tmp_path / "outputs")), _State(500), _Control()
        )
        assert "Warning" in capsys.readouterr().out

    def test_inspect_metadata(self, tmp_path, dirs):
        """inspect() on a saved delta should report correct strategy and tensor count."""
        base_dir, ft_dir = dirs
        out = str(tmp_path / "outputs")
        _make_checkpoint(out, 600, ft_dir)

        DeltaTensorsCallback(base_dir=base_dir, strategy="int4", outlier_fraction=0.01).on_save(
            _Args(out), _State(600), _Control()
        )

        info = dt.inspect(os.path.join(out, "checkpoint-600", "model.wdelta"))
        assert info["strategy"] == "int4"
        assert info["n_tensors"] == len(FT_SD)
        assert len(info["parent_hash"]) == 64  # SHA-256 hex

    def test_wdelta_smaller_than_safetensors(self, tmp_path, dirs):
        """Delta file should be smaller than the original safetensors snapshot."""
        base_dir, ft_dir = dirs
        out = str(tmp_path / "outputs")
        ckpt = _make_checkpoint(out, 700, ft_dir)

        DeltaTensorsCallback(base_dir=base_dir, strategy="sparse", sparsity=0.9).on_save(
            _Args(out), _State(700), _Control()
        )

        sf_size = sum(
            os.path.getsize(os.path.join(ckpt, f))
            for f in os.listdir(ckpt) if f.endswith(".safetensors")
        )
        delta_size = os.path.getsize(os.path.join(ckpt, "model.wdelta"))
        assert delta_size < sf_size
