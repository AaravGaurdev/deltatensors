"""
Apply or swap .wdelta fine-tunes directly in a loaded torch model.

    model = AutoModelForCausalLM.from_pretrained("base", torch_dtype=torch.bfloat16).cuda()
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  # or on CPU

    dt.apply_delta_(model, "math.wdelta")                 # model now = base + math delta
    dt.swap_delta_(model, "code.wdelta", base_state)      # model now = base + code delta

Why swaps restore from base_state instead of subtracting the old delta:
bfloat16 (and float16) addition is not invertible. round(round(w + d) - d)
is often not w, so "subtract the old delta, add the new one" drifts a little
further from the base with every swap. swap_delta_ copies each affected
parameter back from base_state and then adds the new delta, so the result is
always exactly what reconstruct_to_safetensors would produce for that delta.
"""

from __future__ import annotations
import os
import weakref
from collections import OrderedDict
from typing import Dict, Optional, Union

import numpy as np

from . import verify_cache
from .format import WDeltaReader
from .lineage import hash_state_dict
from .reconstruct import _verify_mode, hash_mismatch_error

PathLike = Union[str, os.PathLike]

# names of the parameters the last apply touched, per model (for swap_delta_)
_APPLIED: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
# verified in-memory hashes: fingerprint -> (weakrefs to the tensors, hash).
# The weakrefs guard against freed tensors whose memory (and so data_ptr) was
# reused by new ones, which the CUDA caching allocator does routinely.
_HASHES: "OrderedDict[tuple, tuple]" = OrderedDict()
_MAX_HASHES = 16


def _tensors(model) -> Dict[str, "torch.Tensor"]:  # noqa: F821
    if isinstance(model, dict):
        return model
    return model.state_dict(keep_vars=True)


def _state_fingerprint(state, names) -> tuple:
    """
    Identity + version of every tensor that a hash covers. torch bumps a
    tensor's version on in-place ops, so any normal write changes this.
    (Writes through ``tensor.data`` bypass the version counter; use
    verify="full" if base_state can be modified that way.)
    """
    return tuple((n, state[n].data_ptr(), state[n]._version, tuple(state[n].shape),
                  str(state[n].dtype), str(state[n].device)) for n in names)


def _verify_state(state, names, expected: str, mode: str) -> None:
    if mode == "none":
        return
    names = sorted(names)
    fp = _state_fingerprint(state, names) if mode == "cached" else None
    hit = _HASHES.get(fp) if fp is not None else None
    if hit is not None and all(r() is state[n] for r, n in zip(hit[0], names)):
        actual = hit[1]
    else:
        actual = hash_state_dict({n: state[n] for n in names})  # one tensor on host at a time
        if fp is not None:
            _HASHES[fp] = ([weakref.ref(state[n]) for n in names], actual)
            while len(_HASHES) > _MAX_HASHES:
                _HASHES.popitem(last=False)
    if actual != expected:
        raise hash_mismatch_error(expected, actual)


def _apply(model, wdelta_path: PathLike, base_state, restore: bool, verify) -> list:
    import torch
    from .torch_backend import add_delta_torch_

    mode = _verify_mode(verify)
    params = _tensors(model)

    with WDeltaReader(wdelta_path) as reader:
        if mode == "full":
            reader.verify_checksum()
        elif mode == "cached":
            key = verify_cache.make_key("wdelta-checksum", [wdelta_path])
            if verify_cache.get(key) != "ok":
                reader.verify_checksum()
                verify_cache.put(key, "ok")

        names = sorted(reader.names)
        # validate everything before modifying anything
        missing = [n for n in names if n not in params]
        if missing:
            raise KeyError(f"Tensors not in model: {missing[:10]}")
        if base_state is not None:
            missing = [n for n in names if n not in base_state]
            if missing:
                raise KeyError(f"Tensors not in base_state: {missing[:10]}")
        for n in names:
            shape = list(reader.tensor_metas[n]["shape"])
            if list(params[n].shape) != shape:
                raise ValueError(f"Shape mismatch for '{n}': model {list(params[n].shape)} vs delta {shape}")
            if base_state is not None and list(base_state[n].shape) != shape:
                raise ValueError(f"Shape mismatch for '{n}': base_state {list(base_state[n].shape)} vs delta {shape}")

        _verify_state(base_state if base_state is not None else params, names, reader.parent_hash, mode)

        with torch.no_grad():
            if restore:
                # everything the previous delta touched, plus everything this one will
                prev = _APPLIED.get(model, None) if not isinstance(model, dict) else None
                to_restore = set(names) | (set(prev) if prev is not None else set(base_state))
                for n in sorted(to_restore):
                    if n in params and n in base_state:
                        params[n].copy_(base_state[n])

            seen = set()
            for n in names:
                p = params[n]
                # tied weights (e.g. lm_head / embed_tokens) share storage: add once
                ident = (p.data_ptr(), tuple(p.shape), p.dtype)
                if ident in seen:
                    continue
                seen.add(ident)
                delta = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                add_delta_torch_(reader.payload(n), delta)
                # bf16/f16 params: the add happens in float32 and rounds once,
                # exactly like reconstruct_to_safetensors
                p.add_(delta)
                del delta

    if not isinstance(model, dict):
        _APPLIED[model] = names
    return names


def apply_delta_(model, wdelta_path: PathLike, base_state: Optional[dict] = None,
                 verify: Union[str, bool] = "cached") -> list:
    """
    Add a .wdelta into a torch model's parameters in place, one tensor at a
    time (``param.add_(delta)``), on whatever device each parameter lives on.

    Args:
        model:       torch.nn.Module (or a dict of tensors) whose parameter
                     names match the delta's tensor names.
        wdelta_path: The .wdelta file.
        base_state:  Optional dict of the base weights (CPU or GPU). If given,
                     every parameter the delta touches is first restored from
                     it, so the result doesn't depend on the model's current
                     contents. Otherwise the model must currently hold exactly
                     the base weights.
        verify:      "cached" (default): hash the base weights (the model's, or
                     base_state's) unless the same tensors, unmodified, were
                     already verified in this process. "full": always hash.
                     "none": skip.

    Returns the list of tensor names updated.

    Each updated parameter equals the reconstruction of this delta in the
    parameter's dtype: float32(base) + delta, rounded once. Never undo a delta
    by subtracting it (bf16 addition is not reversible); use swap_delta_.
    """
    return _apply(model, wdelta_path, base_state, restore=base_state is not None, verify=verify)


def swap_delta_(model, new_wdelta_path: PathLike, base_state: dict,
                verify: Union[str, bool] = "cached") -> list:
    """
    Replace whatever delta the model currently carries with ``new_wdelta_path``.

    Every parameter touched by the previously applied delta (as recorded by
    apply_delta_/swap_delta_ on this model; all of base_state if unknown) or by
    the new one is copied back from ``base_state``, then the new delta is
    added. base_state is the source of truth for the base weights and is
    never modified; keep it on the GPU for the fastest swaps.

    The model is only modified after the new delta's base hash has been
    verified against base_state.

    Returns the list of tensor names the new delta updated.
    """
    if base_state is None:
        raise ValueError("swap_delta_ needs base_state: deltas are never undone by subtraction")
    return _apply(model, new_wdelta_path, base_state, restore=True, verify=verify)
