"""
Minimal safetensors reader/writer used by the streaming paths.

The format is an 8-byte little-endian header length, a JSON header mapping
tensor names to {dtype, shape, data_offsets}, then raw little-endian bytes.
Reading it directly (instead of through safetensors/torch) lets us read one
tensor at a time into a preallocated buffer, handle bfloat16 without torch
(as raw uint16), and keep every shard open for the whole pass.
"""

from __future__ import annotations
import json
import os
import re
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# safetensors dtype tag -> (numpy storage dtype, canonical name)
# bfloat16 has no numpy dtype; it is stored and moved around as uint16.
_ST_DTYPES = {
    "F64": (np.float64, "float64"),
    "F32": (np.float32, "float32"),
    "F16": (np.float16, "float16"),
    "BF16": (np.uint16, "bfloat16"),
    "I64": (np.int64, "int64"),
    "I32": (np.int32, "int32"),
    "I16": (np.int16, "int16"),
    "I8": (np.int8, "int8"),
    "U64": (np.uint64, "uint64"),
    "U32": (np.uint32, "uint32"),
    "U16": (np.uint16, "uint16"),
    "U8": (np.uint8, "uint8"),
    "BOOL": (np.bool_, "bool"),
    "F8_E4M3": (np.uint8, "float8_e4m3fn"),
    "F8_E5M2": (np.uint8, "float8_e5m2"),
}
_NAME_TO_TAG = {name: tag for tag, (_, name) in _ST_DTYPES.items()}
FLOAT_DTYPES = ("float64", "float32", "float16", "bfloat16")

_READ_CHUNK = 1 << 30  # FileIO.readinto is capped below 2 GiB per call on Windows


def dtype_name(dtype) -> str:
    """Canonical dtype name for a numpy dtype, torch dtype, or string."""
    s = str(dtype)
    return s[len("torch."):] if s.startswith("torch.") else s


def storage_dtype(name: str) -> np.dtype:
    """numpy dtype used to hold a tensor of dtype ``name`` in memory."""
    return np.dtype(_ST_DTYPES[_NAME_TO_TAG[name]][0])


def parse_size(size: Union[int, str]) -> int:
    """'5GB' -> 5_000_000_000, '500MiB' -> 524_288_000, ints pass through."""
    if isinstance(size, int):
        return size
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(i?)B?\s*", str(size), re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse size {size!r}")
    num, unit, binary = m.groups()
    power = " KMGT".index(unit.upper() or " ")
    return int(float(num) * (1024 if binary else 1000) ** power)


# ---------------------------------------------------------------------------
# dtype conversion
#
# Reconstruction works in one buffer per tensor (see WorkBuffer): the raw base
# tensor is read into the buffer's tail, widened in place to float32 over the
# whole buffer, the delta is added, and the result is narrowed in place into the
# buffer's head. Chunks are converted through a small temporary, front to back;
# with itemsizes <= 4 a chunk's writes only ever land on bytes whose source
# elements were already consumed, so no full-size temporary is needed.
# ---------------------------------------------------------------------------

_CONVERT_CHUNK = 1 << 15  # elements; keeps the chunk temporaries in L2


def _bf16_encode(u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """float32 bits (uint32 chunk) -> bfloat16 bits in t (uint32), RNE like torch."""
    np.right_shift(u, 16, out=t)
    t &= 1
    t += 0x7FFF
    t += u  # NaNs may round into the wrong bits (or wrap); fixed below
    t >>= 16
    # NaN <=> exponent all ones and mantissa nonzero: canonical quiet NaN
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    if nan.any():
        t[nan] = 0x7FC0
    return t


def _convert_chunk(src: np.ndarray, src_dtype: str, dst_dtype: str, tmp32: np.ndarray) -> np.ndarray:
    """Convert one chunk between storage arrays; returns a temporary of dst storage dtype."""
    if src_dtype == "bfloat16":
        t = tmp32.view(np.uint32)[:len(src)]
        np.copyto(t, src, casting="unsafe")
        t <<= 16
        src, src_dtype = t.view(np.float32), "float32"
    if dst_dtype == "bfloat16":
        f = tmp32[:len(src)]
        if src_dtype != "float32" or src is not f:
            np.copyto(f, src, casting="unsafe")
        t = _bf16_encode(f.view(np.uint32).copy(), f.view(np.uint32))
        return t.astype(np.uint16)
    sd = storage_dtype(dst_dtype)
    if np.issubdtype(sd, np.integer) and not np.issubdtype(src.dtype, np.integer):
        src = np.rint(src)
    return src.astype(sd)


def _check_convertible(dtype: str) -> None:
    if dtype.startswith("float8"):
        raise TypeError(f"float8 tensors ({dtype}) cannot be converted")


def convert_inplace(buf: np.ndarray, n: int, src_dtype: str, src_off: int,
                    dst_dtype: str, dst_off: int) -> np.ndarray:
    """
    Convert n elements stored in ``buf`` (a uint8 array) at byte offset
    ``src_off`` to ``dst_dtype`` at byte offset ``dst_off``, front to back.

    Safe when dst_off <= src_off and, for every element i, the destination
    end dst_off + (i+1)*dst_size never passes src_off + (i+1+chunk)*src_size
    over elements not yet read; both layouts used by WorkBuffer satisfy this.
    """
    _check_convertible(src_dtype)
    _check_convertible(dst_dtype)
    s_dt, d_dt = storage_dtype(src_dtype), storage_dtype(dst_dtype)
    src = buf[src_off:src_off + n * s_dt.itemsize].view(s_dt)
    dst = buf[dst_off:dst_off + n * d_dt.itemsize].view(d_dt)
    if src_dtype == dst_dtype and src_off == dst_off:
        return dst
    tmp32 = np.empty(min(_CONVERT_CHUNK, max(n, 1)), dtype=np.float32)
    for s in range(0, n, _CONVERT_CHUNK):
        chunk = _convert_chunk(src[s:s + _CONVERT_CHUNK], src_dtype, dst_dtype, tmp32)
        dst[s:s + len(chunk)] = chunk
    return dst


class WorkBuffer:
    """
    Scratch space for reconstructing one tensor of ``n`` elements stored as
    ``src_dtype``: max(4, itemsize) * n bytes.

        wb = WorkBuffer(n, "bfloat16")
        base.read(name, out=wb.raw)   # raw base bytes, in the tail
        x = wb.widen()                # float32 view over the whole buffer
        x += delta
        out = wb.narrow("bfloat16")   # result view at the head
    """

    def __init__(self, n: int, src_dtype: str):
        self.n = n
        self.src_dtype = src_dtype
        k = storage_dtype(src_dtype).itemsize
        self.buf = np.empty(max(4, k) * n, dtype=np.uint8)
        self._raw_off = len(self.buf) - k * n
        self.raw = self.buf[self._raw_off:].view(storage_dtype(src_dtype))

    def widen(self) -> np.ndarray:
        """Raw tail -> float32 over buf[:4n], in place."""
        return convert_inplace(self.buf, self.n, self.src_dtype, self._raw_off, "float32", 0).view(np.float32)

    def float32(self) -> np.ndarray:
        return self.buf[:4 * self.n].view(np.float32)

    def narrow(self, dst_dtype: str) -> np.ndarray:
        """float32 head -> dst_dtype at the head, in place (float64 gets a new array)."""
        if storage_dtype(dst_dtype).itemsize > 4:
            return self.float32().astype(storage_dtype(dst_dtype))
        return convert_inplace(self.buf, self.n, "float32", 0, dst_dtype, 0)


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """bfloat16 bits (uint16) -> float32, exactly."""
    out = np.empty(raw.shape, dtype=np.float32)
    u = out.view(np.uint32)
    np.copyto(u, raw, casting="unsafe")
    u <<= 16
    return out


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bits (uint16), round-to-nearest-even like torch."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    wb = WorkBuffer(x.size, "float32")
    wb.raw[...] = x.reshape(-1)
    return wb.narrow("bfloat16").copy().reshape(x.shape)


def to_float32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Storage array of dtype ``dtype`` -> float32 (no copy when already float32)."""
    _check_convertible(dtype)
    if dtype == "bfloat16":
        return bf16_to_f32(raw)
    return raw.astype(np.float32, copy=False)


def from_float32(x: np.ndarray, dtype: str) -> np.ndarray:
    """float32 -> storage array of dtype ``dtype`` (no copy when float32)."""
    if dtype == "bfloat16":
        return f32_to_bf16(x)
    sd = storage_dtype(dtype)
    if np.issubdtype(sd, np.integer):
        x = np.rint(x)
    return x.astype(sd, copy=False)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_header(path: Union[str, Path]) -> Tuple[dict, int]:
    """(header dict without __metadata__, byte offset where data starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + n


class TensorInfo:
    __slots__ = ("name", "shard", "dtype", "shape", "start", "nbytes")

    def __init__(self, name, shard, dtype, shape, start, nbytes):
        self.name, self.shard, self.dtype = name, shard, dtype
        self.shape, self.start, self.nbytes = tuple(shape), start, nbytes

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1


class SafetensorsDir:
    """
    Index of every tensor in a folder of .safetensors shards, built once from
    the headers. Shard files are opened lazily and stay open until close().
    ``read(name)`` is thread-safe (one lock per shard).
    """

    def __init__(self, folder: Union[str, Path]):
        self.folder = Path(folder)
        self.shards: List[Path] = sorted(self.folder.glob("*.safetensors"))
        if not self.shards:
            raise FileNotFoundError(f"No .safetensors files in {self.folder}")
        self.tensors: Dict[str, TensorInfo] = {}
        for shard in self.shards:
            header, data_start = read_header(shard)
            for name, meta in header.items():
                if name in self.tensors:
                    raise ValueError(f"Tensor '{name}' appears in more than one shard of {self.folder}")
                tag = meta["dtype"]
                if tag not in _ST_DTYPES:
                    raise TypeError(f"Unsupported safetensors dtype {tag} for '{name}'")
                begin, end = meta["data_offsets"]
                self.tensors[name] = TensorInfo(
                    name, shard, _ST_DTYPES[tag][1], meta["shape"], data_start + begin, end - begin
                )
        self._files: Dict[Path, object] = {}
        self._locks = {s: threading.Lock() for s in self.shards}
        self._open_lock = threading.Lock()

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def keys(self) -> List[str]:
        return sorted(self.tensors)

    def legacy_order(self, keys: List[str]) -> List[str]:
        """deltatensors 0.2.0 hash order: grouped by shard (first appearance), sorted within."""
        groups: Dict[Path, List[str]] = {}
        for k in keys:
            groups.setdefault(self.tensors[k].shard, []).append(k)
        return [k for ks in groups.values() for k in sorted(ks)]

    def _file(self, shard: Path):
        f = self._files.get(shard)
        if f is None:
            with self._open_lock:
                f = self._files.get(shard)
                if f is None:
                    f = self._files[shard] = open(shard, "rb", buffering=0)
        return f

    def read(self, name: str, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Raw storage array (bfloat16 as uint16), shaped. Reads into ``out`` if given."""
        info = self.tensors[name]
        if out is None:
            out = np.empty(info.shape, dtype=storage_dtype(info.dtype))
        buf = memoryview(out.reshape(-1).view(np.uint8))
        if len(buf) != info.nbytes:
            raise ValueError(f"Size mismatch reading '{name}' from {info.shard}")
        f = self._file(info.shard)
        with self._locks[info.shard]:
            f.seek(info.start)
            pos = 0
            while pos < info.nbytes:
                n = f.readinto(buf[pos:pos + _READ_CHUNK])
                if not n:
                    raise ValueError(f"Unexpected end of file reading '{name}' from {info.shard}")
                pos += n
        return out

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __enter__(self) -> "SafetensorsDir":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class ShardedWriter:
    """
    Write tensors, in a fixed order known up front, into one or more
    safetensors shards of at most ``shard_size`` bytes each (a single tensor
    larger than that gets a shard of its own).

    Every shard's header is computed from the plan before any data is written,
    so tensors stream straight to disk. Files are written as ``*.partial`` and
    only renamed into place by ``commit()``; ``abort()`` deletes them.
    """

    def __init__(self, out_dir: Union[str, Path], plan: List[Tuple[str, str, tuple]],
                 shard_size: Union[int, str] = "5GB", metadata: Optional[dict] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        limit = parse_size(shard_size)

        groups: List[List[Tuple[str, str, tuple, int]]] = [[]]
        used = 0
        for name, dtype, shape in plan:
            nbytes = int(np.prod(shape, dtype=np.int64)) * storage_dtype(dtype).itemsize
            if groups[-1] and used + nbytes > limit:
                groups.append([])
                used = 0
            groups[-1].append((name, dtype, tuple(shape), nbytes))
            used += nbytes

        n = len(groups)
        self.files: List[Path] = []
        self._headers: List[bytes] = []
        self._shard_of: Dict[str, int] = {}
        self._expected: List[List[Tuple[str, int]]] = []
        for i, group in enumerate(groups):
            fname = "model.safetensors" if n == 1 else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            self.files.append(self.out_dir / fname)
            header: dict = {"__metadata__": dict(metadata or {"format": "pt"})}
            offset = 0
            for name, dtype, shape, nbytes in group:
                header[name] = {"dtype": _NAME_TO_TAG[dtype], "shape": list(shape),
                                "data_offsets": [offset, offset + nbytes]}
                offset += nbytes
                self._shard_of[name] = i
            hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
            hb += b" " * (-len(hb) % 8)
            self._headers.append(struct.pack("<Q", len(hb)) + hb)
            self._expected.append([(name, nbytes) for name, _, _, nbytes in group])
        self.total_size = sum(nb for g in groups for *_, nb in g)

        self._cur = -1
        self._pos = 0
        self._f = None

    def _partial(self, i: int) -> Path:
        return self.files[i].with_name(self.files[i].name + ".partial")

    def write(self, name: str, arr: np.ndarray) -> None:
        """Append the next tensor in plan order."""
        i = self._shard_of[name]
        if i != self._cur:
            if self._f is not None:
                self._f.close()
            self._cur, self._pos = i, 0
            self._f = open(self._partial(i), "wb")
            self._f.write(self._headers[i])
        exp_name, exp_bytes = self._expected[i][self._pos]
        arr = np.ascontiguousarray(arr)
        if name != exp_name or arr.nbytes != exp_bytes:
            raise ValueError(f"ShardedWriter got '{name}' ({arr.nbytes} B), expected '{exp_name}' ({exp_bytes} B)")
        if arr.nbytes:
            self._f.write(memoryview(arr.reshape(-1).view(np.uint8)))
        self._pos += 1

    def commit(self) -> List[Path]:
        if self._f is not None:
            self._f.close()
            self._f = None
        for i in range(len(self.files)):
            os.replace(self._partial(i), self.files[i])
        if len(self.files) > 1:
            index = {
                "metadata": {"total_size": self.total_size},
                "weight_map": {name: self.files[i].name for name, i in sorted(self._shard_of.items())},
            }
            with open(self.out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
        return list(self.files)

    def abort(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
        for i in range(len(self.files)):
            try:
                os.remove(self._partial(i))
            except FileNotFoundError:
                pass
