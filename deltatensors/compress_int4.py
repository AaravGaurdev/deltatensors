"""
int4 + outlier compression strategy for deltatensors.

Algorithm:
  1. Compute absolute delta magnitudes.
  2. Extract top-k outliers (default: top 1% by magnitude) → stored in float16.
  3. Quantize remaining 99% into 4-bit unsigned integers via asymmetric min-max scaling.
  4. Bit-pack pairs of int4 values into uint8 bytes (true 4-bit storage).

Compression vs quality tradeoff:
  - Outliers in float16: exact preservation of high-signal weights
  - Non-outliers in int4: ~8x size reduction vs float32, small error on low-magnitude values
  - Overall: significantly better signal retention than sparse at similar compression ratios
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any

try:
    import cupy as cp
except ImportError:
    cp = None  # type: ignore


def _xp(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def _to_numpy(arr) -> np.ndarray:
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Bit packing helpers
# ---------------------------------------------------------------------------

def _pack_int4(values) -> np.ndarray:
    """
    Pack an array of int4 values (0-15) into uint8 bytes.
    Two int4 values per byte: high nibble = values[2i], low nibble = values[2i+1].
    Pads with a zero nibble if length is odd. Accepts numpy or cupy; returns numpy.
    """
    xp = _xp(values)
    flat = values.flatten().astype(xp.uint8)
    if len(flat) % 2 != 0:
        flat = xp.concatenate([flat, xp.zeros(1, dtype=xp.uint8)])
    packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
    return _to_numpy(packed.astype(xp.uint8))


def _unpack_int4(packed: np.ndarray, n_elements: int) -> np.ndarray:
    """
    Unpack uint8 bytes back into int4 values (0-15).
    Returns exactly n_elements values.
    """
    high = (packed >> 4) & 0x0F
    low  = packed & 0x0F
    interleaved = np.empty(len(packed) * 2, dtype=np.uint8)
    interleaved[0::2] = high
    interleaved[1::2] = low
    return interleaved[:n_elements]


# ---------------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------------

def compress_int4(
    delta,
    outlier_fraction: float = 0.01,
) -> Dict[str, Any]:
    """
    Compress a delta tensor using outlier extraction + 4-bit quantization.
    Accepts numpy or cupy arrays; always returns numpy arrays.

    Args:
        delta:            Float32 delta array (finetuned - base).
        outlier_fraction: Fraction of weights to store as full-precision outliers.
                          Default 0.01 = top 1% by magnitude.

    Returns:
        Payload dict compatible with deltatensors compress/decompress dispatch.
    """
    xp = _xp(delta)
    orig_shape = delta.shape
    orig_dtype = str(delta.dtype)
    flat = delta.flatten().astype(xp.float32)
    n = len(flat)

    # --- step 1: identify outliers by magnitude ---
    n_outliers = max(1, int(n * outlier_fraction))
    abs_flat = xp.abs(flat)
    outlier_idx = xp.argpartition(abs_flat, -n_outliers)[-n_outliers:]
    outlier_idx = xp.sort(outlier_idx).astype(xp.int64)
    outlier_vals = flat[outlier_idx].astype(xp.float16)

    # --- step 2: mask out outliers for quantization ---
    mask = xp.ones(n, dtype=bool)
    mask[outlier_idx] = False
    non_outlier_vals = flat[mask]  # shape: (n - n_outliers,)

    # --- step 3: asymmetric min-max 4-bit quantization ---
    # float() syncs GPU scalar → Python float (cheap for a single value)
    q_min = float(non_outlier_vals.min())
    q_max = float(non_outlier_vals.max())
    q_range = q_max - q_min

    if q_range < 1e-8:
        scale = np.float16(1.0)
        zero_point = np.float16(q_min)
        quantized = xp.zeros(len(non_outlier_vals), dtype=xp.uint8)
    else:
        scale = np.float16(q_range / 15.0)
        zero_point = np.float16(q_min)
        quantized = xp.clip(
            xp.round((non_outlier_vals - q_min) / q_range * 15.0),
            0, 15
        ).astype(xp.uint8)

    # --- step 4: bit-pack int4 pairs into uint8 ---
    packed = _pack_int4(quantized)  # returns numpy

    return {
        "strategy":         "int4",
        "shape":            list(orig_shape),
        "dtype":            orig_dtype,
        "outlier_fraction": outlier_fraction,
        "n_elements":       n,
        "n_outliers":       n_outliers,
        "outlier_idx":      _to_numpy(outlier_idx),
        "outlier_vals":     _to_numpy(outlier_vals),
        "scale":            np.array([scale], dtype=np.float16),
        "zero_point":       np.array([zero_point], dtype=np.float16),
        "packed":           packed,
    }


# ---------------------------------------------------------------------------
# Decompress
# ---------------------------------------------------------------------------

def decompress_int4(payload: Dict[str, Any]) -> np.ndarray:
    """
    Reconstruct a float32 delta from an int4 + outlier payload.
    """
    n          = payload["n_elements"]
    n_outliers = payload["n_outliers"]
    shape      = payload["shape"]
    dtype      = payload["dtype"]

    outlier_idx  = payload["outlier_idx"]
    outlier_vals = payload["outlier_vals"].astype(np.float32)
    scale        = float(payload["scale"][0])
    zero_point   = float(payload["zero_point"][0])
    packed       = payload["packed"]

    # unpack int4 → float32 for non-outliers
    n_non_outliers = n - n_outliers
    quantized = _unpack_int4(packed, n_non_outliers).astype(np.float32)
    dequantized = quantized / 15.0 * (scale * 15.0) + zero_point  # = q/15 * range + min

    # reconstruct full flat array
    flat = np.empty(n, dtype=np.float32)
    mask = np.ones(n, dtype=bool)
    mask[outlier_idx] = False
    flat[mask] = dequantized
    flat[outlier_idx] = outlier_vals

    return flat.reshape(shape).astype(dtype)