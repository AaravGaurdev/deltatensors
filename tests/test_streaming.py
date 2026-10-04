"""
Streaming reconstruction: reconstruct_to_safetensors, load_delta_from_paths,
multi-shard input/output, worker determinism, and verify modes.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file  # noqa: E402

import deltatensors as dt  # noqa: E402
import deltatensors.reconstruct as rc  # noqa: E402

V1 = Path(__file__).parent / "fixtures" / "v1"
RNG = np.random.default_rng(21)
BASE = {
    "layers.2.w": RNG.standard_normal((48, 40)).astype(np.float32),
    "layers.10.w": RNG.standard_normal((40, 48)).astype(np.float32),
    "layers.3.b": RNG.standard_normal((48,)).astype(np.float32),
    "embed": RNG.standard_normal((300, 16)).astype(np.float32),
    "norm": RNG.standard_normal((7,)).astype(np.float32),
}
FT = {k: v + RNG.standard_normal(v.shape).astype(np.float32) * 0.02 for k, v in BASE.items()}
SHARDS = {"a.safetensors": ["embed", "layers.2.w"], "b.safetensors": ["layers.10.w", "layers.3.b", "norm"]}


def write_dir(path, sd, dtype=torch.bfloat16, shards=SHARDS):
    path.mkdir(parents=True, exist_ok=True)
    for fname, keys in shards.items():
        save_file({k: torch.from_numpy(sd[k]).to(dtype) for k in keys}, str(path / fname))
    return path


@pytest.fixture
def model(tmp_path):
    base = write_dir(tmp_path / "base", BASE)
    (base / "config.json").write_text(json.dumps({"model_type": "toy", "torch_dtype": "float32"}))
    (base / "tokenizer.json").write_text('{"tok": 1}')
    (base / "model.safetensors.index.json").write_text("{}")
    ft = write_dir(tmp_path / "ft", FT)
    delta = tmp_path / "ft.wdelta"
    dt.save_delta_from_paths(delta, ft, base, strategy="int4", use_gpu=False, outlier_fraction=0.02)
    return base, delta


def _as_np(t):
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy()
    return t.numpy()


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_roundtrip_equals_load_delta_from_paths(tmp_path, model, dtype):
    base, delta = model
    out = tmp_path / "out"
    dt.reconstruct_to_safetensors(base, delta, out, dtype=dtype)
    got = load_file(str(out / "model.safetensors"))
    want = dt.load_delta_from_paths(delta, base, dtype=dtype)
    assert set(got) == set(want) == set(BASE)
    for k in BASE:
        assert str(got[k].dtype) == f"torch.{dtype}"
        w = want[k] if dtype == "bfloat16" else torch.from_numpy(want[k])
        np.testing.assert_array_equal(_as_np(got[k]), _as_np(w), err_msg=k)


def test_load_default_dtype_is_exact_float32_of_bf16_result(model):
    base, delta = model
    f32 = dt.load_delta_from_paths(delta, base)
    bf = dt.load_delta_from_paths(delta, base, dtype="bfloat16")
    for k in BASE:
        assert f32[k].dtype == np.float32
        # bf16 output is the float32 result rounded once, not double-rounded
        np.testing.assert_array_equal(_as_np(torch.from_numpy(f32[k]).to(torch.bfloat16)), _as_np(bf[k]))
        assert np.abs(f32[k] - FT[k]).max() < 0.05


def test_float16_base_keeps_dtype(tmp_path):
    base = write_dir(tmp_path / "base", BASE, torch.float16)
    ft = write_dir(tmp_path / "ft", FT, torch.float16)
    delta = tmp_path / "d.wdelta"
    dt.save_delta_from_paths(delta, ft, base, strategy="sparse", sparsity=0.5, use_gpu=False)
    out = dt.load_delta_from_paths(delta, base)
    assert all(v.dtype == np.float16 for v in out.values())


def test_multishard_output_and_model_files(tmp_path, model):
    base, delta = model
    out = tmp_path / "out"
    res = dt.reconstruct_to_safetensors(base, delta, out, shard_size=4000)
    files = sorted(p.name for p in out.glob("*.safetensors"))
    assert len(files) > 2 and files == sorted(Path(f).name for f in res["files"])
    assert files[0].startswith("model-00001-of-")
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(BASE)
    assert index["metadata"]["total_size"] == sum(v.size * 2 for v in BASE.values())
    merged = {}
    for f in files:
        part = load_file(str(out / f))
        assert all(index["weight_map"][k] == f for k in part)
        merged.update(part)
    single = tmp_path / "single"
    dt.reconstruct_to_safetensors(base, delta, single)
    ref = load_file(str(single / "model.safetensors"))
    for k in BASE:
        assert torch.equal(merged[k], ref[k])
    # config/tokenizer copied, dtype updated, old weight index not copied
    assert json.loads((out / "config.json").read_text())["torch_dtype"] == "bfloat16"
    assert (out / "tokenizer.json").read_text() == '{"tok": 1}'
    assert not list(out.glob("*.partial"))


def test_workers_are_deterministic(tmp_path, model):
    base, delta = model
    outs = []
    for w in (1, 2, 5):
        o = tmp_path / f"w{w}"
        dt.reconstruct_to_safetensors(base, delta, o, num_workers=w, shard_size=5000, verify="full")
        outs.append({p.name: p.read_bytes() for p in sorted(o.glob("*.safetensors"))})
    assert outs[0] == outs[1] == outs[2]


def test_passthrough_tensors(tmp_path, model):
    base, delta = model
    extra = {"rotary.inv_freq": torch.arange(5, dtype=torch.float32), "pos_ids": torch.arange(4)}
    save_file(extra, str(base / "c.safetensors"))
    out = tmp_path / "out"
    dt.reconstruct_to_safetensors(base, delta, out, verify="none")
    got = load_file(str(out / "model.safetensors"))
    assert got["rotary.inv_freq"].dtype == torch.bfloat16
    assert torch.equal(got["pos_ids"], extra["pos_ids"])  # integers keep their dtype


# ---------------------------------------------------------------------------
# verify modes
# ---------------------------------------------------------------------------

def _wrong_base(tmp_path):
    other = {k: v + 1.0 for k, v in BASE.items()}
    return write_dir(tmp_path / "wrong", other)


@pytest.mark.parametrize("verify", ["full", "cached", True])
def test_wrong_base_raises_and_leaves_no_output(tmp_path, model, verify):
    _, delta = model
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="hash mismatch"):
        dt.reconstruct_to_safetensors(_wrong_base(tmp_path), delta, out, verify=verify)
    assert not list(out.glob("*.safetensors*"))
    with pytest.raises(ValueError, match="hash mismatch"):
        dt.load_delta_from_paths(delta, _wrong_base(tmp_path / "x"), verify=verify)


@pytest.mark.parametrize("verify", ["none", False])
def test_verify_none_skips(tmp_path, model, verify):
    _, delta = model
    out = dt.load_delta_from_paths(delta, _wrong_base(tmp_path), verify=verify)
    assert set(out) == set(BASE)


class _NoHash:
    """Stands in for hashlib inside reconstruct: any base hashing fails the test."""

    @staticmethod
    def sha256(*a, **k):
        raise AssertionError("base was re-hashed")


def test_cached_skips_rehash_until_a_file_changes(tmp_path, model, monkeypatch):
    base, delta = model
    first = dt.load_delta_from_paths(delta, base)  # populates the cache
    monkeypatch.setattr(rc, "hashlib", _NoHash)
    again = dt.load_delta_from_paths(delta, base)  # cache hit: no hashing
    for k in BASE:
        np.testing.assert_array_equal(first[k], again[k])
    with pytest.raises(AssertionError, match="re-hashed"):
        dt.load_delta_from_paths(delta, base, verify="full")
    monkeypatch.undo()

    # flip one byte of tensor data (same size), keeping the shard valid
    shard = base / "b.safetensors"
    raw = bytearray(shard.read_bytes())
    raw[-3] ^= 0x01
    shard.write_bytes(bytes(raw))
    st = os.stat(shard)
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="hash mismatch"):
        dt.load_delta_from_paths(delta, base)


def test_cached_wrong_hash_fails_fast(tmp_path, model, monkeypatch):
    _, delta = model
    wrong = _wrong_base(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        dt.load_delta_from_paths(delta, wrong)
    monkeypatch.setattr(rc, "hashlib", _NoHash)
    with pytest.raises(ValueError, match="cached result"):
        dt.load_delta_from_paths(delta, wrong)


def test_corrupt_delta_detected(tmp_path, model):
    base, delta = model
    raw = bytearray(delta.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    bad = tmp_path / "bad.wdelta"
    bad.write_bytes(bytes(raw))
    for verify in ("full", "cached"):
        with pytest.raises(ValueError):
            dt.load_delta_from_paths(bad, base, verify=verify)


# ---------------------------------------------------------------------------
# v1 fixtures through the streaming writer
# ---------------------------------------------------------------------------

def test_v1_fixture_to_safetensors(tmp_path):
    with np.load(V1 / "expected.npz") as z:
        want = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith("paths_int4/")}
    out = tmp_path / "out"
    dt.reconstruct_to_safetensors(V1 / "base_bf16", V1 / "paths_int4.wdelta", out, dtype="float32")
    got = load_file(str(out / "model.safetensors"))
    for k, v in want.items():
        np.testing.assert_array_equal(got[k].numpy(), v)
