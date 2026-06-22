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


# ---------------------------------------------------------------------------
# Bit packing helpers
# ---------------------------------------------------------------------------

def _pack_int4(values: np.ndarray) -> np.ndarray:
    """
    Pack an array of int4 values (0-15) into uint8 bytes.
    Two int4 values per byte: high nibble = values[2i], low nibble = values[2i+1].
    Pads with a zero nibble if length is odd.
    """
    flat = values.flatten().astype(np.uint8)
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.uint8(0))  # pad
    packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
    return packed.astype(np.uint8)


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
    delta: np.ndarray,
    outlier_fraction: float = 0.01,
) -> Dict[str, Any]:
    """
    Compress a delta tensor using outlier extraction + 4-bit quantization.

    Args:
        delta:            Float32 delta array (finetuned - base).
        outlier_fraction: Fraction of weights to store as full-precision outliers.
                          Default 0.01 = top 1% by magnitude.

    Returns:
        Payload dict compatible with deltatensors compress/decompress dispatch.
    """
    orig_shape = delta.shape
    orig_dtype = str(delta.dtype)
    flat = delta.flatten().astype(np.float32)
    n = len(flat)

    # --- step 1: identify outliers by magnitude ---
    n_outliers = max(1, int(n * outlier_fraction))
    abs_flat = np.abs(flat)
    # argpartition is O(n) — faster than full sort
    outlier_idx = np.argpartition(abs_flat, -n_outliers)[-n_outliers:]
    outlier_idx = np.sort(outlier_idx).astype(np.int64)
    outlier_vals = flat[outlier_idx].astype(np.float16)

    # --- step 2: mask out outliers for quantization ---
    mask = np.ones(n, dtype=bool)
    mask[outlier_idx] = False
    non_outlier_vals = flat[mask]  # shape: (n - n_outliers,)

    # --- step 3: asymmetric min-max 4-bit quantization ---
    q_min = non_outlier_vals.min()
    q_max = non_outlier_vals.max()
    q_range = q_max - q_min

    if q_range < 1e-8:
        # constant delta — quantize to all zeros
        scale = np.float16(1.0)
        zero_point = np.float16(q_min)
        quantized = np.zeros(len(non_outlier_vals), dtype=np.uint8)
    else:
        scale = np.float16(q_range / 15.0)          # 4-bit: 0..15
        zero_point = np.float16(q_min)
        # clamp to [0, 15] to handle float16 rounding
        quantized = np.clip(
            np.round((non_outlier_vals - q_min) / q_range * 15.0),
            0, 15
        ).astype(np.uint8)

    # --- step 4: bit-pack int4 pairs into uint8 ---
    packed = _pack_int4(quantized)

    return {
        "strategy":         "int4",
        "shape":            list(orig_shape),
        "dtype":            orig_dtype,
        "outlier_fraction": outlier_fraction,
        "n_elements":       n,
        "n_outliers":       n_outliers,
        # outliers
        "outlier_idx":      outlier_idx,
        "outlier_vals":     outlier_vals,
        # non-outlier quantization params
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