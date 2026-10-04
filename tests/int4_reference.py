"""
The 0.2.0 int4 compression and decompression, kept verbatim as the reference
implementation for equivalence tests and benchmarks. v1 layout only.
"""

import numpy as np


def _unpack_int4_reference(packed, n_elements):
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    interleaved = np.empty(len(packed) * 2, dtype=np.uint8)
    interleaved[0::2] = high
    interleaved[1::2] = low
    return interleaved[:n_elements]


def _decompress_int4_reference(payload):
    n = payload["n_elements"]
    n_outliers = payload["n_outliers"]
    shape = payload["shape"]
    dtype = payload["dtype"]

    outlier_idx = payload["outlier_idx"]
    outlier_vals = payload["outlier_vals"].astype(np.float32)
    scale = float(payload["scale"][0])
    zero_point = float(payload["zero_point"][0])
    packed = payload["packed"]

    n_non_outliers = n - n_outliers
    quantized = _unpack_int4_reference(packed, n_non_outliers).astype(np.float32)
    dequantized = quantized / 15.0 * (scale * 15.0) + zero_point

    flat = np.empty(n, dtype=np.float32)
    mask = np.ones(n, dtype=bool)
    mask[outlier_idx] = False
    flat[mask] = dequantized
    flat[outlier_idx] = outlier_vals

    return flat.reshape(shape).astype(dtype)


def _compress_int4_v1_reference(delta, outlier_fraction=0.01):
    """0.2.0 compress_int4: emits the v1 layout (only non-outliers packed)."""
    orig_shape = delta.shape
    orig_dtype = str(delta.dtype)
    flat = delta.flatten().astype(np.float32)
    n = len(flat)

    n_outliers = max(1, int(n * outlier_fraction))
    abs_flat = np.abs(flat)
    outlier_idx = np.argpartition(abs_flat, -n_outliers)[-n_outliers:]
    outlier_idx = np.sort(outlier_idx).astype(np.int64)
    outlier_vals = flat[outlier_idx].astype(np.float16)

    mask = np.ones(n, dtype=bool)
    mask[outlier_idx] = False
    non_outlier_vals = flat[mask]

    q_min = float(non_outlier_vals.min())
    q_max = float(non_outlier_vals.max())
    q_range = q_max - q_min

    if q_range < 1e-8:
        scale = np.float16(1.0)
        zero_point = np.float16(q_min)
        quantized = np.zeros(len(non_outlier_vals), dtype=np.uint8)
    else:
        scale = np.float16(q_range / 15.0)
        zero_point = np.float16(q_min)
        quantized = np.clip(np.round((non_outlier_vals - q_min) / q_range * 15.0), 0, 15).astype(np.uint8)

    q = quantized.astype(np.uint8)
    if len(q) % 2:
        q = np.concatenate([q, np.zeros(1, np.uint8)])
    packed = ((q[0::2] << 4) | (q[1::2] & 0x0F)).astype(np.uint8)

    return {
        "strategy": "int4", "shape": list(orig_shape), "dtype": orig_dtype,
        "outlier_fraction": outlier_fraction, "n_elements": n, "n_outliers": n_outliers,
        "outlier_idx": outlier_idx, "outlier_vals": outlier_vals,
        "scale": np.array([scale], dtype=np.float16),
        "zero_point": np.array([zero_point], dtype=np.float16),
        "packed": packed,
    }
