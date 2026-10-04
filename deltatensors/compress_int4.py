"""
int4 + outlier compression strategy for deltatensors.

Algorithm:
  1. Compute absolute delta magnitudes.
  2. Extract top-k outliers (default: top 1% by magnitude) → stored in float16.
  3. Quantize into 4-bit unsigned integers via asymmetric min-max scaling over
     the non-outlier range (v2: every position is quantized, outliers clipped).
  4. Bit-pack pairs of int4 values into uint8 bytes (true 4-bit storage).

Decoding: delta = q / 15 * (scale * 15) + zero_point, in float32, where scale
and zero_point are the stored float16 values; outlier positions take their
float16 value instead.

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


# ---------------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------------

def n_outliers_for(n: int, outlier_fraction: float) -> int:
    """Number of float16 outliers kept for a tensor of n elements."""
    return min(n, max(1, int(n * outlier_fraction)))


def compress_int4(
    delta,
    outlier_fraction: float = 0.01,
) -> Dict[str, Any]:
    """
    Compress a delta tensor using outlier extraction + 4-bit quantization.
    Accepts numpy or cupy arrays; always returns numpy arrays.

    Emits the v2 layout: all n values are quantized and packed (outlier
    positions are clipped into range; decoding overwrites them), so decoding
    is one full-length pass plus a small scatter, with no boolean mask.
    Quantization targets the float16-rounded scale and zero point that are
    actually stored, so decoding carries no rounding bias from them.

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
    flat = delta.reshape(-1).astype(xp.float32, copy=False)
    n = int(flat.size)

    # --- step 1: identify outliers by magnitude ---
    n_outliers = n_outliers_for(n, outlier_fraction)
    idx_dtype = xp.int32 if n < 2**31 else xp.int64
    outlier_idx = xp.argpartition(xp.abs(flat), n - n_outliers)[n - n_outliers:]
    outlier_idx = xp.sort(outlier_idx).astype(idx_dtype)
    outlier_vals = flat[outlier_idx].astype(xp.float16)

    # --- step 2: range of the non-outliers ---
    mask = xp.ones(n, dtype=bool)
    mask[outlier_idx] = False
    non_outlier_vals = flat[mask]
    if non_outlier_vals.size:
        # float() syncs GPU scalar → Python float (cheap for a single value)
        q_min = float(non_outlier_vals.min())
        q_max = float(non_outlier_vals.max())
    else:
        q_min = q_max = 0.0
    del mask, non_outlier_vals

    # --- step 3: asymmetric min-max 4-bit quantization of all n values ---
    zero_point = np.float16(q_min)
    scale = np.float16((q_max - q_min) / 15.0)
    if q_max - q_min < 1e-8 or float(scale) == 0.0:
        scale = np.float16(1.0)
        quantized = xp.zeros(n, dtype=xp.uint8)
    else:
        quantized = xp.clip(
            xp.rint((flat - float(zero_point)) / float(scale)),
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
        "int4_layout":      2,
        "outlier_idx":      _to_numpy(outlier_idx),
        "outlier_vals":     _to_numpy(outlier_vals),
        "scale":            np.array([scale], dtype=np.float16),
        "zero_point":       np.array([zero_point], dtype=np.float16),
        "packed":           packed,
    }


# ---------------------------------------------------------------------------
# Decompress
# ---------------------------------------------------------------------------

# Bytes gathered per LUT chunk. 16K bytes -> a 128 KB gather buffer that stays
# in L2; measured ~2x faster than 1M-byte chunks on one core.
_CHUNK = 1 << 14


def _pair_lut(scale: float, zero_point: float) -> np.ndarray:
    """
    256 x 2 float32 table: packed byte -> (high-nibble value, low-nibble value).

    Uses the exact float32 op sequence of the original dequant,
    q / 15.0 * (scale * 15.0) + zero_point, so v1 output is bit-identical.
    """
    lut = np.arange(16, dtype=np.float32) / 15.0 * (scale * 15.0) + zero_point
    return np.stack(np.meshgrid(lut, lut, indexing="ij"), -1).reshape(256, 2)


def _lut_add_(packed: np.ndarray, pair: np.ndarray, out: np.ndarray, n_values: int) -> None:
    """out[:n_values] += dequantized values, gathered chunk by chunk via np.take."""
    packed = np.asarray(packed).reshape(-1)
    buf = np.empty((min(_CHUNK, len(packed)), 2), dtype=np.float32)
    for start in range(0, len(packed), _CHUNK):
        chunk = packed[start:start + _CHUNK]
        b = buf[:len(chunk)]
        np.take(pair, chunk, axis=0, out=b)
        vals = b.reshape(-1)
        lo = 2 * start
        hi = min(lo + len(vals), n_values)
        seg = out[lo:hi]
        np.add(seg, vals[:hi - lo], out=seg)


def decompress_int4_add_(payload: Dict[str, Any], out: np.ndarray) -> np.ndarray:
    """
    Add the int4 delta into ``out`` in place (out += delta) and return it.

    ``out`` must be a C-contiguous float32 array with ``n_elements`` elements,
    typically holding the base tensor. The only temporary beyond the bounded
    LUT gather buffer is O(n_outliers) (v2) or O(chunk) (v1).
    """
    n          = payload["n_elements"]
    outlier_idx  = np.asarray(payload["outlier_idx"]).reshape(-1)
    outlier_vals = np.asarray(payload["outlier_vals"]).reshape(-1).astype(np.float32, copy=False)
    pair = _pair_lut(float(payload["scale"][0]), float(payload["zero_point"][0]))

    if out.dtype != np.float32 or not out.flags.c_contiguous or out.size != n:
        raise ValueError("out must be a C-contiguous float32 array with n_elements elements")
    flat = out.reshape(-1)

    if payload.get("int4_layout", 1) >= 2:
        # v2: every position is quantized (outliers clipped). Add everything in
        # one pass, then overwrite outlier positions with base + exact value.
        base_at_idx = flat[outlier_idx]
        _lut_add_(payload["packed"], pair, flat, n)
        flat[outlier_idx] = base_at_idx + outlier_vals
        return out

    # v1: only non-outliers are packed, so output position p holds packed value
    # p - (#outliers before p). Walk the output in chunks: gather that run of
    # packed values, re-insert zeros at the chunk's outlier positions, add.
    packed = np.asarray(payload["packed"]).reshape(-1)
    for a in range(0, n, 2 * _CHUNK):
        b = min(n, a + 2 * _CHUNK)
        ka, kb = np.searchsorted(outlier_idx, [a, b])
        lo, hi = a - ka, b - kb                      # packed value range [lo, hi)
        vals = np.take(pair, packed[lo // 2:(hi + 1) // 2], axis=0).reshape(-1)
        vals = vals[lo % 2:lo % 2 + (hi - lo)]
        if kb > ka:
            local = outlier_idx[ka:kb] - a
            vals = np.insert(vals, local - np.arange(kb - ka, dtype=local.dtype), np.float32(0))
        seg = flat[a:b]
        np.add(seg, vals, out=seg)
    flat[outlier_idx] += outlier_vals
    return out


def decompress_int4(payload: Dict[str, Any]) -> np.ndarray:
    """
    Reconstruct a float32 delta from an int4 + outlier payload.
    """
    out = np.zeros(payload["n_elements"], dtype=np.float32)
    decompress_int4_add_(payload, out)
    return out.reshape(payload["shape"]).astype(payload["dtype"], copy=False)
