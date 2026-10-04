"""
v1 compatibility: .wdelta files written by deltatensors 0.2.0 must keep loading
and give the same output the 0.2.0 reader gave (see fixtures/make_v1_fixtures.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("torch")
pytest.importorskip("safetensors")

import deltatensors as dt  # noqa: E402

V1 = Path(__file__).parent / "fixtures" / "v1"


@pytest.fixture(scope="module")
def expected():
    with np.load(V1 / "expected.npz") as z:
        return {k: z[k] for k in z.files}


def _check(got, expected, prefix):
    want = {k.split("/", 1)[1]: v for k, v in expected.items() if k.startswith(prefix + "/")}
    assert set(got) == set(want)
    for k in want:
        g = np.asarray(got[k], dtype=np.float32)
        np.testing.assert_array_equal(g, want[k], err_msg=k)


@pytest.mark.parametrize("strategy", ["int4", "sparse", "quantized"])
def test_v1_paths(strategy, expected):
    got = dt.load_delta_from_paths(V1 / f"paths_{strategy}.wdelta", V1 / "base_bf16")
    _check(got, expected, f"paths_{strategy}")


def test_v1_chain(expected):
    got = dt.load_delta_chain(
        [V1 / "chain_step1_int4.wdelta", V1 / "chain_step2_int4.wdelta"], V1 / "chain_base_bf16"
    )
    _check(got, expected, "chain")


def test_v1_in_memory(expected):
    with np.load(V1 / "mem_base_f32.npz") as z:
        base = {k: z[k] for k in z.files}
    _check(dt.load_delta(V1 / "mem_int4.wdelta", base), expected, "mem_int4")


def test_v1_inspect():
    info = dt.inspect(V1 / "paths_int4.wdelta")
    assert info["n_tensors"] == 4 and info["strategy"] == "int4"
