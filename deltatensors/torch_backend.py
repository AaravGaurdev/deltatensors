"""
torch implementation of "x += delta" for decoded .wdelta payloads, used for
GPU reconstruction and for applying deltas to a live torch model.

Results are bit-identical to the numpy path: the int4 lookup table is built
in numpy and uploaded, and every step is the same float32 add in the same
order. torch is imported lazily; nothing here is needed for the numpy path.
"""

from __future__ import annotations
from typing import Any, Dict

import numpy as np

from .compress import decompress
from .compress_int4 import _pair_lut

# packed bytes per index_select call: bounds the int64 index + gathered pairs
# (16 bytes per packed byte) to 256 MB of device memory
_TORCH_CHUNK = 1 << 24


def _t(arr: np.ndarray, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


def _lut_add_torch_(packed, pair, out, n_values: int) -> None:
    for s in range(0, packed.numel(), _TORCH_CHUNK):
        vals = pair.index_select(0, packed[s:s + _TORCH_CHUNK].long()).reshape(-1)
        lo = 2 * s
        hi = min(lo + vals.numel(), n_values)
        out[lo:hi].add_(vals[:hi - lo])


def _int4_add_torch_(payload: Dict[str, Any], flat) -> None:
    import torch
    dev = flat.device
    n = payload["n_elements"]
    pair = _t(_pair_lut(float(payload["scale"][0]), float(payload["zero_point"][0])), dev)
    packed = _t(np.asarray(payload["packed"]).reshape(-1), dev)
    idx = _t(np.asarray(payload["outlier_idx"]).reshape(-1).astype(np.int64), dev)
    ovals = _t(np.asarray(payload["outlier_vals"]).reshape(-1).astype(np.float32), dev)

    if payload.get("int4_layout", 1) >= 2:
        base_at_idx = flat[idx]
        _lut_add_torch_(packed, pair, flat, n)
        flat[idx] = base_at_idx + ovals
        return

    # v1: non-outliers only were packed, in order
    dense = torch.zeros(n - idx.numel(), dtype=torch.float32, device=dev)
    _lut_add_torch_(packed, pair, dense, dense.numel())
    mask = torch.ones(n, dtype=torch.bool, device=dev)
    mask[idx] = False
    flat[mask] = flat[mask] + dense
    flat[idx] = flat[idx] + ovals


def add_delta_torch_(payload: Dict[str, Any], x) -> None:
    """
    x += delta in place, where x is a contiguous float32 torch tensor (any
    device) with the payload's element count.
    """
    import torch
    if x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError("x must be a contiguous float32 tensor")
    flat = x.view(-1)
    if flat.numel() != int(np.prod(payload["shape"], dtype=np.int64)):
        raise ValueError(f"size mismatch: tensor has {flat.numel()} elements, delta has shape {payload['shape']}")
    strategy = payload["strategy"]
    if strategy == "int4":
        _int4_add_torch_(payload, flat)
    elif strategy == "sparse":
        idx = _t(np.asarray(payload["indices"]).astype(np.int64), x.device)
        vals = _t(np.asarray(payload["values"]).astype(np.float32), x.device)
        flat[idx] = flat[idx] + vals
    else:
        flat.add_(_t(decompress(payload).astype(np.float32, copy=False).reshape(-1), x.device))


def torch_dtype(name: str):
    import torch
    return getattr(torch, name)


def numpy_to_torch(arr: np.ndarray, dtype: str):
    """Wrap a storage array (bfloat16 as uint16 bits) as a torch tensor, no copy."""
    import torch
    if dtype == "bfloat16":
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)
