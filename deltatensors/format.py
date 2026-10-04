"""
.wdelta (Weight Delta) file format.

Layout on disk:
  [0:7]     Magic bytes: b'wdelta\x00'
  [7:11]    Version:     uint32 little-endian (1 or 2; writers emit 2)
  [11:15]   Header len:  uint32 little-endian (bytes of JSON that follow)
  [15:15+H] JSON header (UTF-8)
  [15+H:-32] Array records, one per (tensor, field). Each record is
             self-describing (tensor name, field, dtype, shape, byte length,
             raw bytes), so records may appear in any order.
  [-32:]    SHA-256 checksum of all preceding bytes

The JSON header contains:
  {
    "parent_hash": "<sha256 hex>",
    "strategy":    "sparse" | "quantized" | "int4",
    "tensors": {
      "<name>": {
        "strategy": ...,
        "shape": [...],
        "dtype": ...,          // dtype of the stored delta (always float32)
        "base_dtype": ...,     // v2: dtype of the base tensor; absent in v1
        // strategy-specific metadata
        // array fields replaced by {"_ref": "<array_key>"}
      }
    }
  }

Version differences:
  v1  int4 packs only the non-outlier values. parent_hash of files written by
      save_delta_from_paths hashes tensors grouped by base shard.
  v2  int4 packs all n values (outlier positions clipped into range, then
      overwritten on decode; tensor field "int4_layout": 2) and stores
      outlier_idx as int32 when n < 2**31. parent_hash always hashes tensors
      in sorted-name order, whatever produced the file.

Readers accept v1 and v2. WDeltaReader parses the header and indexes the
records without reading any array data; each tensor's arrays are then read on
demand.
"""

from __future__ import annotations
import json
import struct
import io
import hashlib
import os
import threading
from typing import Dict, Any, Tuple, Iterator, Mapping, Union
import numpy as np

MAGIC = b"wdelta\x00"
VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
_PREAMBLE = len(MAGIC) + 4 + 4
_ARRAY_FIELDS = {
    "sparse":    ["indices", "values"],
    "quantized": ["scales", "packed_signs"],
    "int4":      ["outlier_idx", "outlier_vals", "scale", "zero_point", "packed"],
}
_HASH_CHUNK = 16 << 20


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


def write_array_record(f, name: str, field: str, arr: np.ndarray) -> None:
    """Serialise one array as a self-describing record."""
    arr = np.ascontiguousarray(arr)
    tn_enc = name.encode("utf-8")
    fl_enc = field.encode("utf-8")
    dt_enc = str(arr.dtype).encode("utf-8")
    f.write(struct.pack("<I", len(tn_enc))); f.write(tn_enc)
    f.write(struct.pack("<I", len(fl_enc))); f.write(fl_enc)
    f.write(struct.pack("<I", len(dt_enc))); f.write(dt_enc)
    f.write(struct.pack("<I", arr.ndim))
    for dim in arr.shape:
        f.write(struct.pack("<Q", dim))
    f.write(struct.pack("<Q", arr.nbytes))
    f.write(memoryview(arr).cast("B") if arr.nbytes else b"")


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

    hw = _HashingWriter(f)
    hw.write(MAGIC)
    hw.write(struct.pack("<I", VERSION))
    hw.write(struct.pack("<I", len(header_bytes)))
    hw.write(header_bytes)
    for (tensor_name, field, arr) in ordered_arrays:
        write_array_record(hw, tensor_name, field, arr)
    f.write(hw.digest())


class _HashingWriter:
    """File-like wrapper that SHA-256s everything written through it."""

    def __init__(self, f):
        self._f = f
        self._h = hashlib.sha256()

    def write(self, b) -> None:
        self._h.update(b)
        self._f.write(b)

    def digest(self) -> bytes:
        return self._h.digest()


def sha256_file_prefix(f, end: int, start: int = 0) -> bytes:
    """SHA-256 of bytes [start, end) of an open binary file, read in chunks."""
    h = hashlib.sha256()
    f.seek(start)
    remaining = end - start
    while remaining > 0:
        chunk = f.read(min(_HASH_CHUNK, remaining))
        if not chunk:
            raise ValueError("Unexpected end of file while checksumming.")
        h.update(chunk)
        remaining -= len(chunk)
    return h.digest()


def append_checksum(path: Union[str, os.PathLike]) -> None:
    """Append the SHA-256 of the whole file as its final 32 bytes."""
    with open(path, "r+b") as f:
        end = f.seek(0, os.SEEK_END)
        digest = sha256_file_prefix(f, end)
        f.seek(end)
        f.write(digest)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class WDeltaReader:
    """
    Lazy .wdelta reader.

    Opening parses the preamble and JSON header and walks the record headers
    (seeking past array data) to build an offset index. Array bytes are only
    read by ``payload(name)``. Safe to call ``payload`` from several threads.

        with WDeltaReader("ft.wdelta") as r:
            r.verify_checksum()
            for name in r.names:
                p = r.payload(name)
    """

    def __init__(self, source: Union[str, os.PathLike, io.IOBase]):
        if isinstance(source, (str, os.PathLike)):
            self._f = open(source, "rb")
            self._owns = True
        else:
            self._f = source
            self._owns = False
        self._lock = threading.Lock()
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    def _parse(self) -> None:
        f = self._f
        start = f.seek(0)
        size = f.seek(0, os.SEEK_END) - start
        f.seek(start)
        if size < _PREAMBLE + 32:
            raise ValueError("Not a .wdelta file (too short).")
        self._start = start
        self._content_end = start + size - 32

        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"Not a .wdelta file (bad magic: {magic!r})")
        self.version = struct.unpack("<I", f.read(4))[0]
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported .wdelta version {self.version} (supported: {SUPPORTED_VERSIONS})"
            )
        header_len = struct.unpack("<I", f.read(4))[0]
        if _PREAMBLE + header_len > size - 32:
            raise ValueError("Corrupted .wdelta header (length exceeds file size).")
        self.header = json.loads(f.read(header_len).decode("utf-8"))
        self.parent_hash = self.header["parent_hash"]
        self.strategy = self.header["strategy"]
        self.tensor_metas: Dict[str, Dict[str, Any]] = self.header["tensors"]
        self.names = list(self.tensor_metas)

        # index the records: (tensor, field) -> (dtype, shape, offset, nbytes)
        self._index: Dict[Tuple[str, str], Tuple[np.dtype, tuple, int, int]] = {}
        pos = start + _PREAMBLE + header_len
        end = self._content_end
        try:
            while pos < end:
                f.seek(pos)
                tn_len = struct.unpack("<I", f.read(4))[0]
                tensor_name = f.read(tn_len).decode("utf-8")
                fl_len = struct.unpack("<I", f.read(4))[0]
                field = f.read(fl_len).decode("utf-8")
                dt_len = struct.unpack("<I", f.read(4))[0]
                dtype = np.dtype(f.read(dt_len).decode("utf-8"))
                ndim = struct.unpack("<I", f.read(4))[0]
                shape = struct.unpack(f"<{ndim}Q", f.read(8 * ndim)) if ndim else ()
                data_len = struct.unpack("<Q", f.read(8))[0]
                offset = f.tell()
                if offset + data_len > end:
                    raise ValueError("record extends past end of file")
                self._index[(tensor_name, field)] = (dtype, shape, offset, data_len)
                pos = offset + data_len
        except (struct.error, UnicodeDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Corrupted .wdelta payload: file may be truncated ({exc}).") from None

    # -- integrity ---------------------------------------------------------

    def verify_checksum(self) -> None:
        """Stream the file through SHA-256 and compare with the trailer."""
        with self._lock:
            actual = sha256_file_prefix(self._f, self._content_end, self._start)
            self._f.seek(self._content_end)
            stored = self._f.read(32)
        if stored != actual:
            raise ValueError("Checksum mismatch: file may be corrupted or truncated.")

    # -- data --------------------------------------------------------------

    def _read_array(self, name: str, field: str) -> np.ndarray:
        try:
            dtype, shape, offset, nbytes = self._index[(name, field)]
        except KeyError:
            raise ValueError(f"Corrupted .wdelta: missing array {field!r} for tensor {name!r}") from None
        buf = bytearray(nbytes)
        with self._lock:
            self._f.seek(offset)
            n = self._f.readinto(buf)
        if n != nbytes:
            raise ValueError("Unexpected end of file reading array data.")
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    def payload(self, name: str) -> Dict[str, Any]:
        """Full payload dict (metadata + numpy arrays) for one tensor."""
        payload = dict(self.tensor_metas[name])
        for field in _ARRAY_FIELDS.get(payload["strategy"], []):
            payload[field] = self._read_array(name, field)
        return payload

    def tensors(self) -> "LazyTensors":
        return LazyTensors(self)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._owns and not self._f.closed:
            self._f.close()

    def __enter__(self) -> "WDeltaReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class LazyTensors(Mapping):
    """Read-only name -> payload mapping that reads each payload on access."""

    def __init__(self, reader: WDeltaReader):
        self._r = reader

    def __getitem__(self, name: str) -> Dict[str, Any]:
        if name not in self._r.tensor_metas:
            raise KeyError(name)
        return self._r.payload(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._r.names)

    def __len__(self) -> int:
        return len(self._r.names)

    def meta(self, name: str) -> Dict[str, Any]:
        """Header metadata only (no array reads)."""
        return self._r.tensor_metas[name]


def read_wdelta(f: io.RawIOBase, verify_checksum: bool = True) -> Tuple[str, str, LazyTensors]:
    """
    Deserialise a .wdelta file.
    Returns (parent_hash, strategy, tensors) where tensors maps tensor name →
    full payload dict with numpy arrays restored. Payloads are read from ``f``
    lazily on access, so ``f`` must stay open while ``tensors`` is used.
    The checksum is verified with a chunked streaming pass (no full-file read).
    """
    reader = WDeltaReader(f)
    if verify_checksum:
        reader.verify_checksum()
    return reader.parent_hash, reader.strategy, reader.tensors()
