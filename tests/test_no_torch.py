"""
The paths API works without torch: safetensors folders are written here with
deltatensors' own writer, with bfloat16 handled as raw bits.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import deltatensors as dt  # noqa: E402
from deltatensors.safetensors_io import ShardedWriter, SafetensorsDir, f32_to_bf16, bf16_to_f32  # noqa: E402
from deltatensors.format import WDeltaReader  # noqa: E402

RNG = np.random.default_rng(5)
BASE = {
    "w1": RNG.standard_normal((33, 20)).astype(np.float32),
    "w2": RNG.standard_normal((20, 33)).astype(np.float32),
    "b": RNG.standard_normal((33,)).astype(np.float32),
}
FT = {k: v + RNG.standard_normal(v.shape).astype(np.float32) * 0.02 for k, v in BASE.items()}


def write_bf16_dir(path, sd, shard_size="1KB"):
    names = sorted(sd)
    w = ShardedWriter(path, [(k, "bfloat16", sd[k].shape) for k in names], shard_size=shard_size)
    for k in names:
        w.write(k, f32_to_bf16(sd[k]))
    w.commit()
    return path


@pytest.mark.parametrize("strategy", ["int4", "sparse", "quantized"])
def test_save_and_reconstruct_without_torch(tmp_path, strategy):
    base = write_bf16_dir(tmp_path / "base", BASE)
    ft = write_bf16_dir(tmp_path / "ft", FT, shard_size="5GB")
    assert len(list(base.glob("*.safetensors"))) > 1
    delta = tmp_path / "d.wdelta"
    dt.save_delta_from_paths(delta, ft, base, strategy=strategy, use_gpu=False)

    with WDeltaReader(delta) as r:
        assert r.version == 2
        assert {m["base_dtype"] for m in r.tensor_metas.values()} == {"bfloat16"}
    assert dt.inspect(delta)["tensors"]["w1"]["dtype"] == "bfloat16"

    got = dt.load_delta_from_paths(delta, base)
    out = tmp_path / "out"
    dt.reconstruct_to_safetensors(base, delta, out)
    with SafetensorsDir(out) as d:
        for k in BASE:
            assert d.tensors[k].dtype == "bfloat16"
            np.testing.assert_array_equal(d.read(k), f32_to_bf16(got[k]))
    if strategy != "quantized":  # BitDelta signs are too coarse for this bound
        for k in BASE:
            assert np.abs(got[k] - bf16_to_f32(f32_to_bf16(FT[k]))).max() < 0.05
