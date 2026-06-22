"""
.wdelta (Weight Delta) file format.

Layout on disk:
  [0:7]     Magic bytes: b'wdelta\x00'
  [7:11]    Version:     uint32 little-endian (currently 1)
  [11:15]   Header len:  uint32 little-endian (bytes of JSON that follow)
  [15:15+H] JSON header (UTF-8)
  [15+H:-32] Binary payload: concatenated numpy arrays in order of tensor index
  [-32:]    SHA-256 checksum of all preceding bytes

The JSON header contains:
  {
    "parent_hash": "<sha256 hex>",
    "strategy":    "sparse" | "quantized" | "int4",
    "tensors": {
      "<name>": {
        "strategy": ...,
        "shape": [...],
        "dtype": ...,
        // strategy-specific metadata
        // array fields replaced by {"_ref": "<array_key>"}
      }
    }
  }

Arrays are serialised in insertion order as raw bytes after the header.
Each array's dtype and shape are stored in the metadata so they can be
reconstructed unambiguously.
"""

from __future__ import annotations
import json
import struct
import io
import hashlib
from typing import Dict, Any, Tuple
import numpy as np

MAGIC = b"wdelta\x00"
VERSION = 1
_ARRAY_FIELDS = {
    "sparse":    ["indices", "values"],
    "quantized": ["scales", "packed_signs"],
    "int4":      ["outlier_idx", "outlier_vals", "scale", "zero_point", "packed"],
}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _extract_arrays(tensor_meta: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Pull numpy arrays out of the payload dict, replace them with {_ref: key}.
    Returns (clean_meta, {key: array}).
    """
    strategy = tensor_meta["strategy"]
    fields = _ARRAY_FIELDS.get(strategy, [])
    arrays = {}
    clean = dict(tensor_meta)
    for field in fields:
        arr = clean.pop(field)
        clean[field] = {"_ref": field}
        arrays[field] = np.asarray(arr)
    return clean, arrays


def write_wdelta(
    f: io.RawIOBase,
    parent_hash: str,
    strategy: str,
    tensors: Dict[str, Dict[str, Any]],
) -> None:
    """
    Serialise a complete delta to a binary file object.
    `tensors` maps tensor name → compress() output dict.
    Appends a SHA-256 checksum of the entire content as the final 32 bytes.
    """
    header = {
        "parent_hash": parent_hash,
        "strategy": strategy,
        "tensors": {},
    }
    ordered_arrays: list[Tuple[str, str, np.ndarray]] = []

    for name, payload in tensors.items():
        clean, arrays = _extract_arrays(payload)
        header["tensors"][name] = clean
        for field, arr in arrays.items():
            ordered_arrays.append((name, field, arr))

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # write into a buffer so we can checksum the whole thing
    buf = io.BytesIO()

    buf.write(MAGIC)
    buf.write(struct.pack("<I", VERSION))
    buf.write(struct.pack("<I", len(header_bytes)))
    buf.write(header_bytes)

    for (tensor_name, field, arr) in ordered_arrays:
        tn_enc = tensor_name.encode("utf-8")
        fl_enc = field.encode("utf-8")
        dt_enc = str(arr.dtype).encode("utf-8")
        buf.write(struct.pack("<I", len(tn_enc))); buf.write(tn_enc)
        buf.write(struct.pack("<I", len(fl_enc))); buf.write(fl_enc)
        buf.write(struct.pack("<I", len(dt_enc))); buf.write(dt_enc)
        buf.write(struct.pack("<I", arr.ndim))
        for dim in arr.shape:
            buf.write(struct.pack("<Q", dim))
        data = arr.tobytes()
        buf.write(struct.pack("<Q", len(data)))
        buf.write(data)

    content = buf.getvalue()
    checksum = hashlib.sha256(content).digest()  # 32 bytes

    f.write(content)
    f.write(checksum)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_wdelta(f: io.RawIOBase) -> Tuple[str, str, Dict[str, Dict[str, Any]]]:
    """
    Deserialise a .wdelta file.
    Returns (parent_hash, strategy, tensors) where tensors maps
    tensor name → full payload dict with numpy arrays restored.
    """
    content = f.read()
    checksum_stored = content[-32:]
    checksum_actual = hashlib.sha256(content[:-32]).digest()
    if checksum_stored != checksum_actual:
        raise ValueError("Checksum mismatch: file may be corrupted or truncated.")

    f = io.BytesIO(content[:-32])  # re-wrap without checksum

    magic = f.read(len(MAGIC))
    if magic != MAGIC:
        raise ValueError(f"Not a .wdelta file (bad magic: {magic!r})")

    version = struct.unpack("<I", f.read(4))[0]
    if version != VERSION:
        raise ValueError(f"Unsupported .wdelta version {version} (expected {VERSION})")

    header_len = struct.unpack("<I", f.read(4))[0]
    header = json.loads(f.read(header_len).decode("utf-8"))

    parent_hash = header["parent_hash"]
    strategy = header["strategy"]
    tensor_metas = header["tensors"]

    array_lookup: Dict[Tuple[str, str], np.ndarray] = {}
    while True:
        chunk = f.read(4)
        if not chunk:
            break
        tn_len = struct.unpack("<I", chunk)[0]
        tensor_name = f.read(tn_len).decode("utf-8")
        fl_len = struct.unpack("<I", f.read(4))[0]
        field = f.read(fl_len).decode("utf-8")
        dt_len = struct.unpack("<I", f.read(4))[0]
        dtype = np.dtype(f.read(dt_len).decode("utf-8"))
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(ndim))
        data_len = struct.unpack("<Q", f.read(8))[0]
        data = f.read(data_len)
        array_lookup[(tensor_name, field)] = np.frombuffer(data, dtype=dtype).reshape(shape)

    tensors: Dict[str, Dict[str, Any]] = {}
    for name, meta in tensor_metas.items():
        payload = dict(meta)
        strat = payload["strategy"]
        for field in _ARRAY_FIELDS.get(strat, []):
            payload[field] = array_lookup[(name, field)]
        tensors[name] = payload

    return parent_hash, strategy, tensors