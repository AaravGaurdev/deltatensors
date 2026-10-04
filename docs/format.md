# Format Spec

## Overview

`.wdelta` is a binary format for storing compressed weight deltas between a base model and a fine-tuned model.

## Layout

```
[0:7]      Magic bytes: b'wdelta\x00'
[7:11]     Version:     uint32 little-endian (1 or 2; current writers emit 2)
[11:15]    Header len:  uint32 little-endian (bytes of JSON that follow)
[15:15+H]  JSON header (UTF-8)
[15+H:-32] Binary payload: one self-describing record per (tensor, array field)
[-32:]     SHA-256 checksum of all preceding bytes
```

## JSON header

```json
{
  "parent_hash": "<sha256 hex of base model>",
  "strategy": "sparse | quantized | int4",
  "tensors": {
    "<tensor_name>": {
      "strategy": "...",
      "shape": [...],
      "dtype": "float32",
      "base_dtype": "bfloat16",
      "<array_field>": {"_ref": "<field_name>"}
    }
  }
}
```

Array fields are replaced with `{"_ref": "<field_name>"}` references — the actual array data lives in the binary payload section.

`dtype` is the dtype the delta is stored and applied in (always `float32`). `base_dtype` (v2 only) is the dtype of the base tensor, e.g. `bfloat16`; reconstruction outputs that dtype unless told otherwise. v1 files have no `base_dtype`, so readers take it from the base model itself.

## Binary payload

Each array is serialised as:

```
[4 bytes]  tensor name length (uint32 LE)
[N bytes]  tensor name (UTF-8)
[4 bytes]  field name length (uint32 LE)
[N bytes]  field name (UTF-8)
[4 bytes]  dtype string length (uint32 LE)
[N bytes]  dtype string (UTF-8, e.g. "float16", "uint8")
[4 bytes]  ndim (uint32 LE)
[8×ndim]   shape dimensions (uint64 LE each)
[8 bytes]  data length in bytes (uint64 LE)
[N bytes]  raw array bytes
```

Records carry their own tensor and field names, so readers must not assume any particular order (0.2.0's streaming writer grouped them by base shard, not by sorted name). Readers index the records once, seeking past the data, and then read each tensor's arrays on demand.

## Compression strategies

### sparse

Keeps the top `(1 - sparsity)` fraction of delta weights by magnitude. Stores indices and values in CSR style.

**Payload arrays:** `indices` (int64), `values` (float32)

**Metadata:** `sparsity`, `shape`, `dtype`

---

### quantized

1-bit sign mask with a learned per-row float16 scale (BitDelta-style).

Reconstructed weight: `scale[row] × sign[row, col]`

**Payload arrays:** `scales` (float16), `packed_signs` (uint8)

**Metadata:** `shape`, `dtype`, `n_elements`, `n_cols`

---

### int4

Outlier extraction + 4-bit quantization:

1. Compute absolute delta magnitudes
2. Extract top-k% outliers → stored as float16
3. Quantize remaining weights into 4-bit unsigned integers via asymmetric min-max scaling
4. Bit-pack pairs of int4 values into uint8 bytes

**Payload arrays:** `outlier_idx` (int32 in v2 when `n_elements < 2**31`, else int64; int64 in v1), `outlier_vals` (float16), `scale` (float16), `zero_point` (float16), `packed` (uint8)

**Metadata:** `shape`, `dtype`, `outlier_fraction`, `n_elements`, `n_outliers`, `int4_layout` (v2: `2`)

**Dequantization:** `value = q / 15 * (scale * 15) + zero_point`, evaluated in float32 with the stored float16 `scale` and `zero_point`. Outlier positions take their float16 value instead.

**Layouts:**

- **v1** (no `int4_layout`): `packed` holds only the `n_elements - n_outliers` non-outlier values, in order. Quantization used the unrounded min/range of the non-outliers.
- **v2** (`int4_layout: 2`): `packed` holds all `n_elements` values. Outlier positions are quantized too (clipped into range) and then overwritten on decode with `base + outlier_val`, so decoding is one full-length pass plus a small scatter, with no boolean mask. Quantization targets the float16-rounded `scale` and `zero_point` that are actually stored. The range is still the min/max of the non-outliers.

## Checksum

The final 32 bytes are a SHA-256 digest of all preceding content. Verified on read — if the file is corrupted or truncated, `read_wdelta` raises `ValueError`.

## parent_hash

SHA-256 over, for each tensor the delta covers, its UTF-8 name followed by its raw stored bytes (bfloat16 as its 16-bit pattern), concatenated:

- **v2:** in sorted-name order, for every writer.
- **v1:** files from `save_delta` used sorted order; files from `save_delta_from_paths` grouped tensors by base shard (shards in order of their first sorted key, names sorted within a shard). Readers verify v1 files in that shard order.

For chained deltas, the hashed model is the float32 reconstruction of the parent.

## Versioning

Writers emit version `2`. Readers accept `1` and `2` and raise `ValueError` on anything else. v1 → v2 changed the int4 layout, the `parent_hash` order of streaming-written files and added `base_dtype`; sparse and quantized payloads are unchanged.

A future v3 is expected to switch `parent_hash` to BLAKE3 over a per-tensor hash tree, so verification can be parallel across tensors.