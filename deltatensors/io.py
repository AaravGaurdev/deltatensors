"""
Public I/O API for deltatensors.

    import deltatensors as dt

    # Save from in-memory state dicts (small models)
    dt.save_delta("checkpoint.wdelta", finetuned, base, strategy="sparse", sparsity=0.9)

    # Save from paths — streaming, O(1) RAM (large models)
    dt.save_delta_from_paths("checkpoint.wdelta", "qwen-finetune/", "qwen-base/", strategy="sparse")

    # Load
    reconstructed = dt.load_delta("checkpoint.wdelta", base)

State dicts can be:
  - Dict[str, np.ndarray]
  - Dict[str, torch.Tensor]   (converted automatically if torch is available)
"""

from __future__ import annotations
import os
import json
import struct
import hashlib
import queue
import contextlib
import threading
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

from .compress import compress, decompress
from .format import (
    write_wdelta, WDeltaReader, MAGIC, VERSION, _ARRAY_FIELDS,
    write_array_record, append_checksum,
)
from .compress_int4 import n_outliers_for
from .lineage import hash_state_dict, verify_base
from .safetensors_io import SafetensorsDir, to_float32, dtype_name
from .reconstruct import load_delta_from_paths, reconstruct_to_safetensors, _verify_mode  # noqa: F401

StateDict = Dict[str, Union[np.ndarray, "torch.Tensor"]]  # noqa: F821

_SENTINEL = object()  # signals producer is done

_GPU_UNAVAILABLE_REASON: str = ""
try:
    import cupy as cp
    _n = cp.cuda.runtime.getDeviceCount()
    if _n > 0:
        _HAS_GPU = True
    else:
        _HAS_GPU = False
        _GPU_UNAVAILABLE_REASON = "no CUDA devices found (getDeviceCount() == 0)"
except ImportError:
    cp = None  # type: ignore
    _HAS_GPU = False
    _GPU_UNAVAILABLE_REASON = "cupy not installed"
except Exception as _e:
    cp = None  # type: ignore
    _HAS_GPU = False
    _GPU_UNAVAILABLE_REASON = f"cupy init error: {_e}"

# tensors larger than this go to CPU to avoid VRAM OOM from intermediate arrays
# int4 peak ≈ 17× element count; 50M elements = ~850 MB peak on a typical 12 GB card
_GPU_ELEMENT_THRESHOLD = 50_000_000


def _to_numpy(state_dict: StateDict) -> Dict[str, np.ndarray]:
    """numpy arrays for math; torch bfloat16 tensors become float32 (exactly)."""
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = v
        else:
            try:
                t = v.detach().cpu()
            except AttributeError:
                raise TypeError(f"Cannot convert tensor '{k}' of type {type(v)} to numpy.")
            out[k] = t.float().numpy() if str(t.dtype) == "torch.bfloat16" else t.numpy()
    return out


def _check_same_keys(a: set, b: set, a_name: str, b_name: str) -> None:
    if a != b:
        msg = f"Key mismatch between {a_name} and {b_name}."
        if a - b:
            msg += f"\n  Only in {a_name}: {sorted(a - b)}"
        if b - a:
            msg += f"\n  Only in {b_name}: {sorted(b - a)}"
        raise ValueError(msg)


def _raw_bytes(arr: np.ndarray) -> memoryview:
    return memoryview(np.ascontiguousarray(arr).reshape(-1).view(np.uint8))


def _tensor_header_entry(strategy: str, shape: list, kwargs: dict, base_dtype: Optional[str] = None) -> dict:
    """
    Build the per-tensor JSON header entry (scalars + _ref placeholders) from shape alone.
    Mirrors what compress() returns, minus the array fields — so the header can be written
    before any tensor data is loaded.
    """
    n = int(np.prod(shape)) if shape else 1
    entry: dict = {"strategy": strategy, "shape": shape, "dtype": "float32"}
    if base_dtype is not None:
        entry["base_dtype"] = base_dtype

    if strategy == "sparse":
        entry["sparsity"] = kwargs.get("sparsity", 0.9)
    elif strategy == "quantized":
        n_cols = int(np.prod(shape[1:])) if len(shape) > 1 else n
        entry["n_elements"] = n
        entry["n_cols"] = n_cols
    elif strategy == "int4":
        outlier_fraction = kwargs.get("outlier_fraction", 0.01)
        entry["outlier_fraction"] = outlier_fraction
        entry["n_elements"] = n
        entry["n_outliers"] = n_outliers_for(n, outlier_fraction)
        entry["int4_layout"] = 2

    for field in _ARRAY_FIELDS.get(strategy, []):
        entry[field] = {"_ref": field}

    return entry


_write_array_to_file = write_array_record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_delta(
    path: Union[str, Path],
    finetuned: StateDict,
    base: StateDict,
    strategy: str = "sparse",
    use_gpu: bool = True,
    **kwargs,
) -> str:
    """
    Compute and save the delta between `finetuned` and `base` to `path`.
    Loads both models fully into RAM. For large models (>3B), use save_delta_from_paths.

    Args:
        path:      Output file path (conventionally *.wdelta).
        finetuned: State dict of the fine-tuned model.
        base:      State dict of the base model.
        strategy:  "sparse" or "quantized".
        **kwargs:  Strategy-specific options (e.g. sparsity=0.9 for sparse).

    Returns:
        The SHA-256 hash of the base model (for lineage tracking).
    """
    ft_np = _to_numpy(finetuned)
    base_np = _to_numpy(base)

    ft_keys = set(ft_np.keys())
    base_keys = set(base_np.keys())
    if ft_keys != base_keys:
        only_ft = ft_keys - base_keys
        only_base = base_keys - ft_keys
        msg = "Key mismatch between finetuned and base state dicts."
        if only_ft:
            msg += f"\n  Only in finetuned: {sorted(only_ft)}"
        if only_base:
            msg += f"\n  Only in base:      {sorted(only_base)}"
        raise ValueError(msg)

    parent_hash = hash_state_dict(base)
    _use_gpu = _HAS_GPU and use_gpu

    compressed_tensors = {}
    for name in sorted(ft_np.keys()):
        ft_arr = ft_np[name].astype(np.float32)
        base_arr = base_np[name].astype(np.float32)
        if ft_arr.shape != base_arr.shape:
            raise ValueError(
                f"Shape mismatch for '{name}': finetuned {ft_arr.shape} vs base {base_arr.shape}. "
                f"Architecture mutations are not supported in v0.1."
            )
        delta = ft_arr - base_arr
        if _use_gpu:
            delta = cp.asarray(delta)
        compressed_tensors[name] = compress(delta, strategy, **kwargs)
        compressed_tensors[name]["base_dtype"] = dtype_name(base[name].dtype)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        write_wdelta(f, parent_hash, strategy, compressed_tensors)

    size_mb = os.path.getsize(path) / 1e6
    print(f"[deltatensors] saved {path.name}  ({len(compressed_tensors)} tensors, {size_mb:.1f} MB, strategy={strategy})")
    return parent_hash


def save_delta_from_paths(
    out_path: Union[str, Path],
    finetuned_dir: Union[str, Path],
    base_dir: Union[str, Path],
    strategy: str = "sparse",
    prefetch: int = 2,
    use_gpu: bool = True,
    **kwargs,
) -> str:
    """
    Streaming delta save — peak RAM is O(prefetch tensors), not O(two full models).

    Architecture:
      - Pass 1: read safetensors metadata headers (no tensor data) to build the output
        header and write it immediately — no need to buffer compressed results first.
      - Pass 2: producer thread reads tensor pairs; consumer compresses and pushes
        arrays onto a bounded write queue; writer thread drains the queue to disk
        concurrently. Compress and write phases overlap.
      - parent_hash (SHA-256 of base) is filled in as a placeholder and seek-patched
        after streaming; the file checksum is computed with one final read pass.

    Args:
        out_path:      Output .wdelta file path.
        finetuned_dir: Folder containing finetuned safetensors shards.
        base_dir:      Folder containing base safetensors shards.
        strategy:      "sparse", "quantized", or "int4".
        prefetch:      Bound on the read queue and write queue (default 2).
        **kwargs:      Strategy-specific options.

    Returns:
        The SHA-256 hash of the base model.
    """
    with SafetensorsDir(finetuned_dir) as ft, SafetensorsDir(base_dir) as base:
        return _save_delta_from_dirs(out_path, ft, base, strategy, prefetch, use_gpu, kwargs)


def _save_delta_from_dirs(out_path, ft: SafetensorsDir, base: SafetensorsDir,
                          strategy: str, prefetch: int, use_gpu: bool, kwargs: dict) -> str:
    _use_gpu = _HAS_GPU and use_gpu

    _check_same_keys(set(ft.tensors), set(base.tensors), "finetuned", "base")
    all_keys = ft.keys()
    for name in all_keys:
        if ft.tensors[name].shape != base.tensors[name].shape:
            raise ValueError(f"Shape mismatch for '{name}': finetuned {list(ft.tensors[name].shape)} "
                             f"vs base {list(base.tensors[name].shape)}.")
    if _use_gpu:
        _device = "GPU"
    else:
        _device = f"CPU ({_GPU_UNAVAILABLE_REASON})" if _GPU_UNAVAILABLE_REASON else "CPU"
    print(f"[deltatensors] streaming {len(all_keys)} tensors (strategy={strategy}, prefetch={prefetch}, device={_device})...")

    # --- pass 1: build header from safetensors metadata (zero tensor I/O) ---
    _PLACEHOLDER_HASH = "0" * 64  # SHA-256 hex is always exactly 64 ASCII chars
    header = {
        "parent_hash": _PLACEHOLDER_HASH,
        "strategy": strategy,
        "tensors": {
            name: _tensor_header_entry(strategy, list(ft.tensors[name].shape), kwargs,
                                       base.tensors[name].dtype)
            for name in all_keys
        },
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # locate placeholder in the file so we can seek-patch it after streaming
    ph_offset_in_header = header_bytes.index(_PLACEHOLDER_HASH.encode("ascii"))
    parent_hash_file_offset = len(MAGIC) + 4 + 4 + ph_offset_in_header  # magic+ver+hdrlen

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- write header, then stream arrays via a writer thread ---
    fields = _ARRAY_FIELDS.get(strategy, [])
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)

        write_queue: queue.Queue = queue.Queue(maxsize=prefetch)
        writer_error: list = [None]

        def writer():
            try:
                while True:
                    item = write_queue.get()
                    if item is _SENTINEL:
                        break
                    wname, wfield, warr = item
                    _write_array_to_file(f, wname, wfield, warr)
            except Exception as exc:
                writer_error[0] = exc

        t_writer = threading.Thread(target=writer, daemon=True)
        t_writer.start()

        read_queue: queue.Queue = queue.Queue(maxsize=prefetch)
        producer_error: list = [None]
        base_hasher = hashlib.sha256()

        def producer():
            try:
                # tensors are visited in sorted-name order, which is the order
                # parent_hash covers (format v2); shard handles stay open
                for name in all_keys:
                    raw = base.read(name)
                    base_arr = to_float32(raw, base.tensors[name].dtype)
                    ft_arr = to_float32(ft.read(name), ft.tensors[name].dtype)
                    read_queue.put((name, raw, base_arr, ft_arr))
            except Exception as exc:
                producer_error[0] = exc
            finally:
                read_queue.put(_SENTINEL)

        t_producer = threading.Thread(target=producer, daemon=True)
        t_producer.start()

        count = 0
        try:
            while True:
                item = read_queue.get()
                if item is _SENTINEL:
                    break
                if producer_error[0]:
                    raise producer_error[0]
                if writer_error[0]:
                    raise writer_error[0]

                name, base_raw, base_arr, ft_arr = item
                base_hasher.update(name.encode("utf-8"))
                base_hasher.update(_raw_bytes(base_raw))
                del base_raw

                delta = ft_arr.astype(np.float32) - base_arr.astype(np.float32)
                del base_arr, ft_arr

                # skip GPU for large tensors (embeddings etc.) to avoid VRAM OOM;
                # compress_int4 creates ~5 intermediate arrays so peak ≈ 17× element count
                if _use_gpu and delta.size <= _GPU_ELEMENT_THRESHOLD:
                    delta = cp.asarray(delta)

                compressed = compress(delta, strategy, **kwargs)
                del delta

                for field in fields:
                    write_queue.put((name, field, np.asarray(compressed[field])))
                del compressed

                count += 1
                if count % 50 == 0:
                    print(f"[deltatensors]   {count}/{len(all_keys)} tensors compressed...")
        finally:
            write_queue.put(_SENTINEL)

        t_producer.join()
        t_writer.join()

        if producer_error[0]:
            raise producer_error[0]
        if writer_error[0]:
            raise writer_error[0]

    parent_hash = base_hasher.hexdigest()

    # seek-patch the placeholder hash, then append the file checksum
    with open(out_path, "r+b") as f:
        f.seek(parent_hash_file_offset)
        f.write(parent_hash.encode("ascii"))
    append_checksum(out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[deltatensors] saved {out_path.name}  ({len(all_keys)} tensors, {size_mb:.1f} MB)")
    return parent_hash


def load_delta(
    path: Union[str, Path],
    base: StateDict,
    verify: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct a fine-tuned model from a .wdelta file and a base state dict.

    Args:
        path:   Path to the .wdelta file.
        base:   State dict of the base model.
        verify: SHA-256 verify the base before reconstructing (recommended).

    Returns:
        Reconstructed state dict as Dict[str, np.ndarray].
    """
    base_np = _to_numpy(base)

    with WDeltaReader(path) as reader:
        reader.verify_checksum()
        parent_hash, strategy = reader.parent_hash, reader.strategy
        if verify:
            verify_base(base, parent_hash)

        reconstructed = {}
        for name in reader.names:
            if name not in base_np:
                raise KeyError(f"Tensor '{name}' not found in base model.")
            payload = reader.payload(name)
            delta = decompress(payload)
            base_arr = base_np[name].astype(np.float32)
            reconstructed[name] = (base_arr + delta).astype(payload["dtype"])

    print(f"[deltatensors] loaded {Path(path).name}  ({len(reconstructed)} tensors, strategy={strategy})")
    return reconstructed

def inspect(path: Union[str, Path]) -> dict:
    """
    Return metadata from a .wdelta file without loading the base model.
    """
    with WDeltaReader(path) as reader:
        reader.verify_checksum()

    size_mb = os.path.getsize(path) / 1e6
    return {
        "path": str(path),
        "size_mb": round(size_mb, 2),
        "version": reader.version,
        "parent_hash": reader.parent_hash,
        "strategy": reader.strategy,
        "n_tensors": len(reader.names),
        "tensors": {
            name: {"shape": meta["shape"], "dtype": meta.get("base_dtype", meta["dtype"])}
            for name, meta in reader.tensor_metas.items()
        },
    }


def inspect_chain(delta_paths: list) -> list:
    """
    Return metadata for each step in a lineage chain without loading any tensors.

    The ``parent_hash`` field in each entry is the SHA-256 of the model that step
    was computed against.  For a valid chain, step N's parent_hash should equal the
    SHA-256 of the reconstructed model produced by step N-1.  This cannot be verified
    without actually loading models; use ``load_delta_chain(..., verify=True)`` for that.

    Args:
        delta_paths: Ordered list of .wdelta paths, oldest first.

    Returns:
        List of dicts (one per file), each with the fields from ``inspect()`` plus
        a ``"step"`` key (0-indexed).

    Example::

        history = dt.inspect_chain(["v1.wdelta", "v2.wdelta", "v3.wdelta"])
        for entry in history:
            print(entry["step"], entry["size_mb"], "MB", entry["parent_hash"][:8])
    """
    return [dict(inspect(p), step=i) for i, p in enumerate(delta_paths)]


def load_delta_chain(
    delta_paths: list,
    base: Union[str, Path, "StateDict"],
    verify: Union[str, bool] = "cached",
    num_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct the final model by applying a sequence of delta files in order.

    Chain layout::

        base ──► delta_paths[0] ──► model_1 ──► delta_paths[1] ──► model_2 ──► …

    Each delta is applied to the model produced by the prior step.  With
    ``verify=True``, the ``parent_hash`` of each delta is checked against the
    SHA-256 of the current model, catching any out-of-order or wrong-base error.

    Args:
        delta_paths: Ordered list of .wdelta paths, oldest first.
        base:        Base state dict **or** path to a base safetensors directory.
                     Passing a directory uses the streaming loader for the first
                     step, which keeps RAM proportional to one model, not two.
        verify:      "cached" (default), "full" or "none" for the base folder (see
                     load_delta_from_paths); later steps are always hashed
                     in memory unless verify is "none"/False.
        num_workers: Threads for the streaming first step.

    Returns:
        Reconstructed state dict at the end of the chain.

    Example::

        sd = dt.load_delta_chain(
            ["finetune_v1.wdelta", "finetune_v2.wdelta"],
            base="path/to/base-model",
        )
    """
    if not delta_paths:
        raise ValueError("delta_paths is empty — provide at least one .wdelta file.")

    delta_paths = [str(p) for p in delta_paths]

    mode = _verify_mode(verify)

    if isinstance(base, (str, Path)):
        # streaming first step: avoids loading base fully into RAM. Chain hashes
        # are over float32 models, so intermediate results stay float32.
        current: Dict[str, np.ndarray] = load_delta_from_paths(
            delta_paths[0], base, verify=mode, dtype="float32", num_workers=num_workers,
        )
        remaining = delta_paths[1:]
    else:
        current = _to_numpy(base)
        remaining = delta_paths

    for path in remaining:
        current = load_delta(path, current, verify=mode != "none")

    n = len(delta_paths)
    print(f"[deltatensors] chain: applied {n} delta(s) successfully")
    return current


def save_delta_chain_from_paths(
    out_path: Union[str, Path],
    finetuned_dir: Union[str, Path],
    parent_delta_path: Union[str, Path],
    base_dir: Union[str, Path],
    strategy: str = "sparse",
    use_gpu: bool = True,
    **kwargs,
) -> str:
    """
    Save a new chained delta — the difference between ``finetuned_dir`` and the
    model reconstructed from ``parent_delta_path``.

    Use this when each fine-tune step is a small increment on top of the previous
    one (e.g. continual learning, multi-stage RLHF).  The resulting chain of
    .wdelta files encodes the full training trajectory::

        base ──► parent_delta_path ──► parent_model ──► [out_path] ──► finetuned

    Streaming implementation — peak RAM is O(one tensor pair), not O(two full models).
    The parent ``.wdelta`` arrays are read sequentially from disk one tensor at a time;
    the finetuned and base tensors are loaded on demand and freed immediately.

    Args:
        out_path:            Output .wdelta file for the new delta.
        finetuned_dir:       Folder containing the new finetuned model safetensors.
        parent_delta_path:   Path to the immediately prior .wdelta in the chain.
        base_dir:            Folder containing the original base model safetensors
                             (needed to reconstruct the parent model tensor-by-tensor).
        strategy:            Compression strategy: ``"sparse"``, ``"quantized"``,
                             or ``"int4"``.
        use_gpu:             Use GPU (CuPy) for compression if available.
        **kwargs:            Strategy-specific options.

    Returns:
        The SHA-256 hash of the parent model (stored as ``parent_hash`` in the new
        .wdelta file — the link that ties this delta into the chain).

    Example::

        # chain: base → v1 → v2
        dt.save_delta_chain_from_paths(
            "v2.wdelta",
            finetuned_dir="v2_checkpoint/",
            parent_delta_path="v1.wdelta",
            base_dir="base_model/",
            strategy="int4",
            outlier_fraction=0.05,
        )
    """
    with SafetensorsDir(finetuned_dir) as ft, SafetensorsDir(base_dir) as base:
        return _save_chain_from_dirs(out_path, ft, str(parent_delta_path), base, strategy, use_gpu, kwargs)


def _save_chain_from_dirs(out_path, ft: SafetensorsDir, parent_delta_path: str, base: SafetensorsDir,
                          strategy: str, use_gpu: bool, kwargs: dict) -> str:
    out_path = Path(out_path)
    _use_gpu = _HAS_GPU and use_gpu

    print(f"[deltatensors] chain: streaming delta vs {Path(parent_delta_path).name}...")

    # Index the parent .wdelta (header + record offsets only) and verify its
    # checksum with a streaming pass; arrays are then read by tensor name.
    with WDeltaReader(parent_delta_path) as parent:
        parent.verify_checksum()
        all_keys = sorted(parent.names)

        _check_same_keys(set(all_keys), set(ft.tensors), "parent delta", "finetuned model")
        missing = [k for k in all_keys if k not in base]
        if missing:
            raise KeyError(f"Tensors not found in base model: {missing[:10]}")

        print(f"[deltatensors] chain: {len(all_keys)} tensors (strategy={strategy})...")

        # Build output header from finetuned safetensors metadata (zero tensor I/O)
        _PLACEHOLDER_HASH = "0" * 64
        out_header = {
            "parent_hash": _PLACEHOLDER_HASH,
            "strategy": strategy,
            "tensors": {
                name: _tensor_header_entry(strategy, list(ft.tensors[name].shape), kwargs,
                                           base.tensors[name].dtype)
                for name in all_keys
            },
        }
        out_header_bytes = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
        ph_offset = out_header_bytes.index(_PLACEHOLDER_HASH.encode("ascii"))
        parent_hash_file_offset = len(MAGIC) + 4 + 4 + ph_offset

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fields = _ARRAY_FIELDS.get(strategy, [])
        parent_hasher = hashlib.sha256()

        with open(out_path, "wb") as out_f:
            out_f.write(MAGIC)
            out_f.write(struct.pack("<I", VERSION))
            out_f.write(struct.pack("<I", len(out_header_bytes)))
            out_f.write(out_header_bytes)

            for i, name in enumerate(all_keys):
                # Reconstruct parent tensor = base_tensor + decompress(parent_delta)
                base_arr = to_float32(base.read(name), base.tensors[name].dtype)
                parent_arr = (base_arr + decompress(parent.payload(name))).astype(np.float32)
                del base_arr

                # Accumulate hash of the reconstructed parent (becomes parent_hash in output)
                parent_hasher.update(name.encode("utf-8"))
                parent_hasher.update(parent_arr.tobytes())

                # Load finetuned tensor and compute delta
                ft_arr = to_float32(ft.read(name), ft.tensors[name].dtype)
                delta = ft_arr.astype(np.float32) - parent_arr
                del ft_arr, parent_arr

                if _use_gpu and delta.size <= _GPU_ELEMENT_THRESHOLD:
                    delta = cp.asarray(delta)

                compressed = compress(delta, strategy, **kwargs)
                del delta

                for field in out_fields:
                    _write_array_to_file(out_f, name, field, np.asarray(compressed[field]))
                del compressed

                if (i + 1) % 50 == 0:
                    print(f"[deltatensors] chain:   {i + 1}/{len(all_keys)} tensors compressed...")

    parent_hash = parent_hasher.hexdigest()

    # Seek-patch the placeholder hash, then append the file checksum
    with open(out_path, "r+b") as out_f:
        out_f.seek(parent_hash_file_offset)
        out_f.write(parent_hash.encode("ascii"))
    append_checksum(out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[deltatensors] saved {out_path.name}  ({len(all_keys)} tensors, {size_mb:.1f} MB)")
    return parent_hash