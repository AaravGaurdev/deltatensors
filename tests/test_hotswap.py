"""
apply_delta_ / swap_delta_ on live torch models.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file  # noqa: E402

import deltatensors as dt  # noqa: E402
import deltatensors.hotswap as hs  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class Toy(torch.nn.Module):
    def __init__(self, tie=False):
        super().__init__()
        self.embed = torch.nn.Embedding(50, 24)
        self.proj = torch.nn.Linear(24, 24)
        self.head = torch.nn.Linear(24, 50, bias=False)
        if tie:
            self.head.weight = self.embed.weight


def _sd(seed, scale):
    g = torch.Generator().manual_seed(seed)
    return {
        "embed.weight": torch.randn(50, 24, generator=g) * scale,
        "proj.weight": torch.randn(24, 24, generator=g) * scale,
        "proj.bias": torch.randn(24, generator=g) * scale,
        "head.weight": torch.randn(50, 24, generator=g) * scale,
    }


@pytest.fixture(scope="module")
def deltas(tmp_path_factory):
    """base dir + three fine-tunes (A, B, and C touching only proj.*)."""
    root = tmp_path_factory.mktemp("hs")
    base = {k: v.to(torch.bfloat16) for k, v in _sd(0, 1.0).items()}
    fts = {
        "A": {k: (base[k].float() + v).to(torch.bfloat16) for k, v in _sd(1, 0.02).items()},
        "B": {k: (base[k].float() + v).to(torch.bfloat16) for k, v in _sd(2, 0.03).items()},
    }
    (root / "base").mkdir()
    save_file(base, str(root / "base" / "model.safetensors"))
    paths = {}
    for name, sd in fts.items():
        (root / name).mkdir()
        save_file(sd, str(root / name / "model.safetensors"))
        paths[name] = root / f"{name}.wdelta"
        dt.save_delta_from_paths(paths[name], root / name, root / "base", strategy="int4", use_gpu=False)
    # C: only proj.* (in-memory save over a subset of tensors)
    sub = ["proj.bias", "proj.weight"]
    paths["C"] = root / "C.wdelta"
    dt.save_delta(paths["C"], {k: fts["B"][k] for k in sub}, {k: base[k] for k in sub},
                  strategy="sparse", sparsity=0.5, use_gpu=False)
    expected = {}
    for name in ("A", "B"):
        out = root / f"out_{name}"
        dt.reconstruct_to_safetensors(root / "base", paths[name], out, dtype="bfloat16")
        expected[name] = load_file(str(out / "model.safetensors"))
    return base, paths, expected


def _model(base, device, tie=False):
    m = Toy(tie=tie).to(torch.bfloat16).to(device)
    with torch.no_grad():
        for k, p in m.state_dict(keep_vars=True).items():
            p.copy_(base[k])
    return m


def _assert_model_equals(m, want):
    sd = m.state_dict()
    for k, v in want.items():
        assert torch.equal(sd[k].cpu(), v), k


@pytest.mark.parametrize("device", DEVICES)
def test_apply_matches_reconstruct(deltas, device):
    base, paths, expected = deltas
    m = _model(base, device)
    names = dt.apply_delta_(m, paths["A"])
    assert names == sorted(expected["A"])
    _assert_model_equals(m, expected["A"])


@pytest.mark.parametrize("device", DEVICES)
def test_swap_cycles_never_drift(deltas, device):
    base, paths, expected = deltas
    base_state = {k: v.to(device) for k, v in base.items()}
    m = _model(base, device)
    dt.apply_delta_(m, paths["A"])
    for _ in range(5):
        dt.swap_delta_(m, paths["B"], base_state)
        _assert_model_equals(m, expected["B"])
        dt.swap_delta_(m, paths["A"], base_state)
        _assert_model_equals(m, expected["A"])
    for k in base:
        assert torch.equal(base_state[k].cpu(), base[k])  # never modified


def test_subtracting_a_delta_would_drift(deltas):
    """Why swap_delta_ restores from base_state: bf16 add/sub doesn't round-trip."""
    base, paths, _ = deltas
    m = _model(base, "cpu")
    w0 = m.proj.weight.detach().clone()
    delta = torch.zeros(w0.shape)
    with dt.format.WDeltaReader(paths["A"]) as r:
        dt.torch_backend.add_delta_torch_(r.payload("proj.weight"), delta)
    with torch.no_grad():
        m.proj.weight.add_(delta)
        m.proj.weight.sub_(delta)
    assert not torch.equal(m.proj.weight, w0)


def test_double_apply_is_refused(deltas):
    base, paths, expected = deltas
    m = _model(base, "cpu")
    dt.apply_delta_(m, paths["A"])
    with pytest.raises(ValueError, match="hash mismatch"):
        dt.apply_delta_(m, paths["A"])  # model no longer holds the base
    _assert_model_equals(m, expected["A"])


def test_wrong_base_state_leaves_model_untouched(deltas):
    base, paths, expected = deltas
    m = _model(base, "cpu")
    dt.apply_delta_(m, paths["A"])
    wrong = {k: v + 1 for k, v in base.items()}
    with pytest.raises(ValueError, match="hash mismatch"):
        dt.swap_delta_(m, paths["B"], wrong)
    _assert_model_equals(m, expected["A"])


def test_swap_to_partial_delta_restores_previous_tensors(deltas):
    base, paths, expected = deltas
    m = _model(base, "cpu")
    dt.apply_delta_(m, paths["A"])
    dt.swap_delta_(m, paths["C"], base)
    sd = m.state_dict()
    for k in ("embed.weight", "head.weight"):  # touched by A, not by C
        assert torch.equal(sd[k], base[k])
    assert not torch.equal(sd["proj.weight"], base["proj.weight"])


def test_swap_needs_base_state(deltas):
    _, paths, _ = deltas
    with pytest.raises(ValueError, match="base_state"):
        dt.swap_delta_(Toy(), paths["A"], None)


def test_tied_weights_applied_once(tmp_path):
    base = {k: v.to(torch.bfloat16) for k, v in _sd(0, 1.0).items()}
    base["head.weight"] = base["embed.weight"]
    ft = {k: (v.float() + 0.01 * torch.ones_like(v.float())).to(torch.bfloat16) for k, v in base.items()}
    ft["head.weight"] = ft["embed.weight"]
    d = tmp_path / "t.wdelta"
    dt.save_delta(d, ft, base, strategy="sparse", sparsity=0.0, use_gpu=False)
    m = _model(base, "cpu", tie=True)
    dt.apply_delta_(m, d)
    assert m.head.weight is m.embed.weight
    assert torch.equal(m.embed.weight, ft["embed.weight"])


def test_cached_verification_rehashes_after_inplace_change(deltas, monkeypatch):
    base, paths, expected = deltas
    base_state = {k: v.clone() for k, v in base.items()}
    m = _model(base, "cpu")
    calls = []
    real = hs.hash_state_dict
    monkeypatch.setattr(hs, "hash_state_dict", lambda sd: calls.append(1) or real(sd))
    dt.swap_delta_(m, paths["A"], base_state)
    dt.swap_delta_(m, paths["B"], base_state)
    dt.swap_delta_(m, paths["A"], base_state)
    assert len(calls) == 1  # verified once, then cached
    with torch.no_grad():
        base_state["proj.bias"].add_(1.0)  # in-place edit bumps the tensor version
    with pytest.raises(ValueError, match="hash mismatch"):
        dt.swap_delta_(m, paths["B"], base_state)
    assert len(calls) == 2
    _assert_model_equals(m, expected["A"])
