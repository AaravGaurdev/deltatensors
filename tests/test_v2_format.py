"""
Format v2: writer version, sorted-order parent_hash, and multi-shard layouts
that 0.2.0 got wrong.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("torch")
pytest.importorskip("safetensors")

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import deltatensors as dt  # noqa: E402
from deltatensors.format import WDeltaReader, read_wdelta, write_wdelta, VERSION  # noqa: E402
from deltatensors.compress import compress  # noqa: E402

RNG = np.random.default_rng(11)
# "layers.10" sorts before "layers.2": keys interleave across shards
BASE = {
    "layers.2.w": RNG.standard_normal((40, 24)).astype(np.float32),
    "layers.10.w": RNG.standard_normal((24, 40)).astype(np.float32),
    "layers.3.b": RNG.standard_normal((40,)).astype(np.float32),
    "embed": RNG.standard_normal((50, 8)).astype(np.float32),
}
FT1 = {k: v + RNG.standard_normal(v.shape).astype(np.float32) * 0.02 for k, v in BASE.items()}
FT2 = {k: v + RNG.standard_normal(v.shape).astype(np.float32) * 0.01 for k, v in FT1.items()}

SHARDS_A = {"a-1.safetensors": ["embed", "layers.2.w"], "a-2.safetensors": ["layers.10.w", "layers.3.b"]}
SHARDS_B = {"b-1.safetensors": ["layers.10.w", "layers.2.w"], "b-2.safetensors": ["embed"],
            "b-3.safetensors": ["layers.3.b"]}


def write_dir(path, sd, shards, dtype=torch.float32):
    path.mkdir(parents=True, exist_ok=True)
    for fname, keys in shards.items():
        save_file({k: torch.from_numpy(sd[k]).to(dtype) for k in keys}, str(path / fname))
    return path


def test_writer_emits_v2():
    buf = io.BytesIO()
    write_wdelta(buf, "0" * 64, "int4", {"w": compress(BASE["embed"], "int4")})
    buf.seek(0)
    assert WDeltaReader(buf).version == VERSION == 2


def test_rejects_unknown_version():
    buf = io.BytesIO()
    write_wdelta(buf, "0" * 64, "sparse", {"w": compress(BASE["embed"], "sparse")})
    raw = bytearray(buf.getvalue())
    raw[7:11] = (99).to_bytes(4, "little")
    with pytest.raises(ValueError, match="Unsupported .wdelta version 99"):
        read_wdelta(io.BytesIO(bytes(raw)))


def test_lazy_reader_reads_nothing_until_asked(tmp_path):
    p = tmp_path / "x.wdelta"
    dt.save_delta(p, FT1, BASE, strategy="int4", use_gpu=False)
    with WDeltaReader(p) as r:
        assert r.names == sorted(BASE)
        r.verify_checksum()
        payload = r.payload("embed")
        assert payload["packed"].dtype == np.uint8


@pytest.mark.parametrize("ft_shards", [SHARDS_A, SHARDS_B], ids=["same-layout", "different-layout"])
def test_paths_hash_matches_in_memory_hash(tmp_path, ft_shards):
    base_dir = write_dir(tmp_path / "base", BASE, SHARDS_A)
    ft_dir = write_dir(tmp_path / "ft", FT1, ft_shards)
    p = tmp_path / "d.wdelta"
    h = dt.save_delta_from_paths(p, ft_dir, base_dir, strategy="int4", use_gpu=False)
    # v2: the streaming writer and hash_state_dict agree (sorted-name order)
    assert h == dt.hash_state_dict(BASE)
    # so a paths-written delta verifies against an in-memory base, and vice versa
    a = dt.load_delta(p, BASE)
    b = dt.load_delta_from_paths(p, base_dir)
    for k in BASE:
        np.testing.assert_array_equal(a[k], b[k])
        assert np.abs(a[k] - FT1[k]).max() < 0.02


def test_chain_on_interleaved_multishard_parent(tmp_path):
    # 0.2.0 crashed: it read parent records assuming sorted order
    base_dir = write_dir(tmp_path / "base", BASE, SHARDS_A)
    ft1_dir = write_dir(tmp_path / "ft1", FT1, SHARDS_A)
    ft2_dir = write_dir(tmp_path / "ft2", FT2, SHARDS_B)
    d1, d2 = tmp_path / "v1.wdelta", tmp_path / "v2.wdelta"
    dt.save_delta_from_paths(d1, ft1_dir, base_dir, strategy="int4", use_gpu=False, outlier_fraction=0.05)
    dt.save_delta_chain_from_paths(d2, ft2_dir, d1, base_dir, strategy="int4", use_gpu=False,
                                   outlier_fraction=0.05)
    out = dt.load_delta_chain([d1, d2], base_dir)
    for k in BASE:
        assert np.abs(out[k] - FT2[k]).max() < 0.02


def test_in_memory_bf16_hash_matches_paths_hash(tmp_path):
    bf = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in BASE.items()}
    ft = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in FT1.items()}
    base_dir = write_dir(tmp_path / "base", BASE, SHARDS_A, torch.bfloat16)
    ft_dir = write_dir(tmp_path / "ft", FT1, SHARDS_B, torch.bfloat16)
    h_paths = dt.save_delta_from_paths(tmp_path / "p.wdelta", ft_dir, base_dir, strategy="int4", use_gpu=False)
    h_mem = dt.save_delta(tmp_path / "m.wdelta", ft, bf, strategy="int4", use_gpu=False)
    assert h_paths == h_mem == dt.hash_state_dict(bf)
    # each file verifies against the other representation of the base
    dt.load_delta(tmp_path / "p.wdelta", bf)
    dt.load_delta_from_paths(tmp_path / "m.wdelta", base_dir, verify="full")
    assert dt.inspect(tmp_path / "m.wdelta")["tensors"]["embed"]["dtype"] == "bfloat16"
