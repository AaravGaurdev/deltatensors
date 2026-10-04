"""
Generate the v1 .wdelta fixtures used by tests/test_v1_compat.py.

These files were produced by deltatensors 0.2.0 (commit 8c605a9), BEFORE the
v2 format existed. Do not regenerate them with a newer release: their whole
point is to pin what an old writer emitted.

    python tests/fixtures/make_v1_fixtures.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

import deltatensors as dt  # noqa: E402

OUT = HERE / "v1"


def _sd(rng, scale=1.0):
    # odd element counts exercise int4 nibble padding; "layers.10" < "layers.2"
    # in sorted order, so keys interleave across the two shards below
    return {
        "layers.2.w":  rng.standard_normal((37, 19)).astype(np.float32) * scale,
        "layers.10.w": rng.standard_normal((48, 24)).astype(np.float32) * scale,
        "layers.2.b":  rng.standard_normal((37,)).astype(np.float32) * scale,
        "embed":       rng.standard_normal((101, 16)).astype(np.float32) * scale,
    }


SHARDS = {
    "model-00001-of-00002.safetensors": ["embed", "layers.2.w", "layers.2.b"],
    "model-00002-of-00002.safetensors": ["layers.10.w"],
}


def _write_dir(path, sd, dtype, shards=None):
    path.mkdir(parents=True, exist_ok=True)
    for fname, keys in (shards or SHARDS).items():
        save_file({k: torch.from_numpy(sd[k]).to(dtype) for k in keys}, str(path / fname))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(1234)

    base = _sd(rng)
    ft1 = {k: v + rng.standard_normal(v.shape).astype(np.float32) * 0.02 for k, v in base.items()}
    ft2 = {k: v + rng.standard_normal(v.shape).astype(np.float32) * 0.01 for k, v in ft1.items()}

    _write_dir(OUT / "base_bf16", base, torch.bfloat16)
    _write_dir(OUT / "ft1_bf16", ft1, torch.bfloat16)
    _write_dir(OUT / "ft2_bf16", ft2, torch.bfloat16)

    expected = {}
    for strat, kw in [("int4", {"outlier_fraction": 0.05}), ("sparse", {"sparsity": 0.7}), ("quantized", {})]:
        p = OUT / f"paths_{strat}.wdelta"
        dt.save_delta_from_paths(p, OUT / "ft1_bf16", OUT / "base_bf16", strategy=strat, use_gpu=False, **kw)
        for k, v in dt.load_delta_from_paths(p, OUT / "base_bf16").items():
            expected[f"paths_{strat}/{k}"] = v

    # chain: base -> ft1 (int4) -> ft2 (int4). Single-shard dirs: 0.2.0's chain
    # writer reads parent records assuming sorted order, which a multi-shard
    # interleaved parent violates (it crashes), so the fixture can't use one.
    one = {"model.safetensors": sorted(base)}
    _write_dir(OUT / "chain_base_bf16", base, torch.bfloat16, one)
    _write_dir(OUT / "chain_ft1_bf16", ft1, torch.bfloat16, one)
    _write_dir(OUT / "chain_ft2_bf16", ft2, torch.bfloat16, one)
    step1 = OUT / "chain_step1_int4.wdelta"
    chain = OUT / "chain_step2_int4.wdelta"
    dt.save_delta_from_paths(step1, OUT / "chain_ft1_bf16", OUT / "chain_base_bf16",
                             strategy="int4", use_gpu=False, outlier_fraction=0.05)
    dt.save_delta_chain_from_paths(chain, OUT / "chain_ft2_bf16", step1, OUT / "chain_base_bf16",
                                   strategy="int4", use_gpu=False, outlier_fraction=0.05)
    for k, v in dt.load_delta_chain([step1, chain], OUT / "chain_base_bf16").items():
        expected[f"chain/{k}"] = v

    # in-memory float32 path
    np.savez(OUT / "mem_base_f32.npz", **base)
    mem = OUT / "mem_int4.wdelta"
    dt.save_delta(mem, ft1, base, strategy="int4", use_gpu=False, outlier_fraction=0.01)
    for k, v in dt.load_delta(mem, base).items():
        expected[f"mem_int4/{k}"] = v

    np.savez(OUT / "expected.npz", **expected)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
