"""
The torch backend (GPU reconstruction, hot-swap) must be bit-identical to the
numpy reference path.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

import deltatensors as dt  # noqa: E402
from deltatensors.compress import compress, decompress_add_  # noqa: E402
import deltatensors.torch_backend as tb  # noqa: E402
from int4_reference import _compress_int4_v1_reference  # noqa: E402

V1 = Path(__file__).parent / "fixtures" / "v1"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _payloads(shape, seed):
    rng = np.random.default_rng(seed)
    d = (rng.standard_normal(shape) * 0.01).astype(np.float32)
    return [
        compress(d, "int4", outlier_fraction=0.03),
        _compress_int4_v1_reference(d, outlier_fraction=0.03),
        compress(d, "sparse", sparsity=0.8),
        compress(d, "quantized"),
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(1, 2), (7,), (33, 17), (300, 257)])
@pytest.mark.parametrize("chunk", [3, 1 << 24])
def test_add_delta_torch_matches_numpy(monkeypatch, device, shape, chunk):
    monkeypatch.setattr(tb, "_TORCH_CHUNK", chunk)
    base = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    for p in _payloads(shape, 2):
        want = base.copy()
        decompress_add_(p, want)
        x = torch.from_numpy(base.copy()).to(device)
        tb.add_delta_torch_(p, x)
        np.testing.assert_array_equal(x.cpu().numpy(), want, err_msg=p["strategy"])


@needs_cuda
@pytest.mark.parametrize("fixture", ["paths_int4", "paths_sparse", "paths_quantized"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_cuda_reconstruct_matches_cpu(tmp_path, fixture, dtype):
    outs = {}
    for device in ("cpu", "cuda"):
        o = tmp_path / device
        dt.reconstruct_to_safetensors(V1 / "base_bf16", V1 / f"{fixture}.wdelta", o,
                                      dtype=dtype, device=device, num_workers=3)
        outs[device] = (o / "model.safetensors").read_bytes()
    assert outs["cpu"] == outs["cuda"]


@needs_cuda
def test_cuda_load_delta_from_paths_v2(tmp_path):
    from safetensors.torch import save_file
    rng = np.random.default_rng(3)
    base = {"a": rng.standard_normal((512, 300)).astype(np.float32), "b": rng.standard_normal(77).astype(np.float32)}
    ft = {k: v + rng.standard_normal(v.shape).astype(np.float32) * 0.01 for k, v in base.items()}
    for name, sd in (("base", base), ("ft", ft)):
        (tmp_path / name).mkdir()
        save_file({k: torch.from_numpy(v).to(torch.bfloat16) for k, v in sd.items()},
                  str(tmp_path / name / "model.safetensors"))
    d = tmp_path / "d.wdelta"
    dt.save_delta_from_paths(d, tmp_path / "ft", tmp_path / "base", strategy="int4", use_gpu=False)
    cpu = dt.load_delta_from_paths(d, tmp_path / "base")
    gpu = dt.load_delta_from_paths(d, tmp_path / "base", device="cuda")
    for k in base:
        np.testing.assert_array_equal(cpu[k], gpu[k])
