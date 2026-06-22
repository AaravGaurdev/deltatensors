"""
Compression strategies for delta weights.

sparse:     zero out the smallest-magnitude deltas (CSR-style storage)
quantized:  1-bit sign mask + per-row float16 scale (BitDelta-style)
int4:       outlier extraction (float16) + 4-bit quantization for remainder
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Sparse
# ---------------------------------------------------------------------------

def compress_sparse(delta: np.ndarray, sparsity: float) -> Dict[str, Any]:
    """
    Keep only the top-(1-sparsity) fraction of delta weights by magnitude.
    Stores (indices, values, shape) — enough to reconstruct the full matrix.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    flat = delta.flatten()
    k = max(1, int(len(flat) * (1.0 - sparsity)))
    threshold_idx = np.argpartition(np.abs(flat), -k)[-k:]
    indices = np.sort(threshold_idx).astype(np.int64)
    values = flat[indices].astype(np.float32)

    return {
        "strategy": "sparse",
        "shape": list(delta.shape),
        "dtype": str(delta.dtype),
        "sparsity": sparsity,
        "indices": indices,
        "values": values,
    }


def decompress_sparse(payload: Dict[str, Any]) -> np.ndarray:
    flat = np.zeros(int(np.prod(payload["shape"])), dtype=np.float32)
    flat[payload["indices"]] = payload["values"]
    return flat.reshape(payload["shape"]).astype(payload["dtype"])


# ---------------------------------------------------------------------------
# Quantized (BitDelta-style)
# ---------------------------------------------------------------------------

def compress_quantized(delta: np.ndarray) -> Dict[str, Any]:
    """
    1-bit sign mask with a learned per-row float16 scale.
    Reconstructed weight: scale[row] * sign[row, col]
    """
    orig_shape = delta.shape
    orig_dtype = str(delta.dtype)

    mat = delta.reshape(delta.shape[0], -1).astype(np.float32) if delta.ndim > 1 else delta.reshape(1, -1).astype(np.float32)

    signs = np.sign(mat).astype(np.int8)
    signs[signs == 0] = 1

    scales = np.mean(np.abs(mat), axis=1).astype(np.float16)

    sign_bits = (signs > 0).astype(np.uint8)
    packed = np.packbits(sign_bits.flatten())

    return {
        "strategy": "quantized",
        "shape": list(orig_shape),
        "dtype": orig_dtype,
        "scales": scales,
        "packed_signs": packed,
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