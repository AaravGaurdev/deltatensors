"""
Compression strategies for delta weights.

sparse:     zero out the smallest-magnitude deltas (CSR-style storage)
quantized:  1-bit sign mask + per-row float16 scale (BitDelta-style)
int4:       outlier extraction (float16) + 4-bit quantization for remainder
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any

try:
    import cupy as cp
except ImportError:
    cp = None  # type: ignore


def _xp(arr):
    """Return cupy if arr lives on GPU, else numpy."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def _to_numpy(arr) -> np.ndarray:
    """Move array to CPU numpy (no-op if already numpy)."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Sparse
# ---------------------------------------------------------------------------

def compress_sparse(delta, sparsity: float) -> Dict[str, Any]:
    """
    Keep only the top-(1-sparsity) fraction of delta weights by magnitude.
    Stores (indices, values, shape) — enough to reconstruct the full matrix.
    Accepts numpy or cupy arrays; always returns numpy arrays.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    xp = _xp(delta)
    flat = delta.flatten()
    k = max(1, int(len(flat) * (1.0 - sparsity)))
    threshold_idx = xp.argpartition(xp.abs(flat), -k)[-k:]
    indices = xp.sort(threshold_idx).astype(xp.int64)
    values = flat[indices].astype(xp.float32)

    return {
        "strategy": "sparse",
        "shape": list(delta.shape),
        "dtype": str(delta.dtype),
        "sparsity": sparsity,
        "indices": _to_numpy(indices),
        "values": _to_numpy(values),
    }


def decompress_sparse(payload: Dict[str, Any]) -> np.ndarray:
    flat = np.zeros(int(np.prod(payload["shape"])), dtype=np.float32)
    flat[payload["indices"]] = payload["values"]
    return flat.reshape(payload["shape"]).astype(payload["dtype"])


# ---------------------------------------------------------------------------
# Quantized (BitDelta-style)
# ---------------------------------------------------------------------------

def compress_quantized(delta) -> Dict[str, Any]:
    """
    1-bit sign mask with a learned per-row float16 scale.
    Reconstructed weight: scale[row] * sign[row, col]
    Accepts numpy or cupy arrays; always returns numpy arrays.
    """
    orig_shape = delta.shape
    orig_dtype = str(delta.dtype)
    xp = _xp(delta)

    mat = (delta.reshape(delta.shape[0], -1).astype(xp.float32)
           if delta.ndim > 1
           else delta.reshape(1, -1).astype(xp.float32))

    signs = xp.sign(mat).astype(xp.int8)
    signs[signs == 0] = 1

    scales = xp.mean(xp.abs(mat), axis=1).astype(xp.float16)

    sign_bits = (signs > 0).astype(xp.uint8)
    packed = xp.packbits(sign_bits.flatten())

    return {
        "strategy": "quantized",
        "shape": list(orig_shape),
        "dtype": orig_dtype,
        "scales": _to_numpy(scales),
        "packed_signs": _to_numpy(packed),
        "n_elements": int(mat.shape[0] * mat.shape[1]),
        "n_cols": int(mat.shape[1]),
    }


def decompress_quantized(payload: Dict[str, Any]) -> np.ndarray:
    n_elements = payload["n_elements"]
    n_cols = payload["n_cols"]
    n_rows = n_elements // n_cols

    unpacked = np.unpackbits(payload["packed_signs"])[:n_elements]
    signs = unpacked.reshape(n_rows, n_cols).astype(np.float32)
    signs[signs == 0] = -1.0

    scales = payload["scales"].astype(np.float32)
    mat = signs * scales[:, np.newaxis]

    return mat.reshape(payload["shape"]).astype(payload["dtype"])


# ---------------------------------------------------------------------------
# int4 + outlier (imported from submodule)
# ---------------------------------------------------------------------------

def compress_int4(delta: np.ndarray, outlier_fraction: float = 0.01) -> Dict[str, Any]:
    from .compress_int4 import compress_int4 as _compress_int4
    return _compress_int4(delta, outlier_fraction=outlier_fraction)


def decompress_int4(payload: Dict[str, Any]) -> np.ndarray:
    from .compress_int4 import decompress_int4 as _decompress_int4
    return _decompress_int4(payload)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compress(delta: np.ndarray, strategy: str, **kwargs) -> Dict[str, Any]:
    if strategy == "sparse":
        sparsity = kwargs.get("sparsity", 0.9)
        return compress_sparse(delta, sparsity)
    elif strategy == "quantized":
        return compress_quantized(delta)
    elif strategy == "int4":
        outlier_fraction = kwargs.get("outlier_fraction", 0.01)
        return compress_int4(delta, outlier_fraction=outlier_fraction)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'sparse', 'quantized', or 'int4'.")


_ROW_CHUNK_ELEMS = 1 << 20


def _sparse_add_(payload: Dict[str, Any], flat: np.ndarray) -> None:
    flat[payload["indices"]] += np.asarray(payload["values"]).astype(np.float32, copy=False)


def _quantized_add_(payload: Dict[str, Any], flat: np.ndarray) -> None:
    n_elements = payload["n_elements"]
    n_cols = payload["n_cols"]
    n_rows = n_elements // n_cols
    mat = flat.reshape(n_rows, n_cols)
    scales = payload["scales"].astype(np.float32)
    packed = payload["packed_signs"]
    # rows per chunk, keeping the per-chunk temporaries around _ROW_CHUNK_ELEMS
    step = max(1, _ROW_CHUNK_ELEMS // max(1, n_cols))
    for r0 in range(0, n_rows, step):
        r1 = min(n_rows, r0 + step)
        start, count = r0 * n_cols, (r1 - r0) * n_cols
        b0 = start // 8
        bits = np.unpackbits(packed[b0:(start + count + 7) // 8])
        bits = bits[start - 8 * b0:start - 8 * b0 + count].reshape(r1 - r0, n_cols)
        s = scales[r0:r1, np.newaxis]
        # same float32 values as decompress_quantized: sign * scale
        mat[r0:r1] += np.where(bits.astype(bool), s, -s)


def decompress_add_(payload: Dict[str, Any], out: np.ndarray) -> np.ndarray:
    """
    out += delta, in place, for any strategy. ``out`` must be a C-contiguous
    float32 array with the tensor's element count (typically the base tensor).
    Bit-identical to ``out + decompress(payload)``.
    """
    if out.dtype != np.float32 or not out.flags.c_contiguous:
        raise ValueError("out must be a C-contiguous float32 array")
    strategy = payload["strategy"]
    if strategy == "int4":
        from .compress_int4 import decompress_int4_add_
        decompress_int4_add_(payload, out)
        return out
    flat = out.reshape(-1)
    if strategy == "sparse":
        _sparse_add_(payload, flat)
    elif strategy == "quantized":
        _quantized_add_(payload, flat)
    else:
        raise ValueError(f"Unknown strategy '{strategy}' in payload.")
    return out


def decompress(payload: Dict[str, Any]) -> np.ndarray:
    strategy = payload["strategy"]
    if strategy == "sparse":
        return decompress_sparse(payload)
    elif strategy == "quantized":
        return decompress_quantized(payload)
    elif strategy == "int4":
        return decompress_int4(payload)
    else:
        raise ValueError(f"Unknown strategy '{strategy}' in payload.")