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
import threading
from pathlib import Path
from typing import Dict, Union
import numpy as np

from .compress import compress, decompress
from .format import write_wdelta, read_wdelta, MAGIC, VERSION, _ARRAY_FIELDS
from .lineage import hash_state_dict, verify_base

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
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = v
        else:
            try:
                out[k] = v.detach().cpu().numpy()
            except AttributeError:
                raise TypeError(f"Cannot convert tensor '{k}' of type {type(v)} to numpy.")
    return out


def _safetensors_keys(folder: str) -> list[str]:
    import torch
    try:
        
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install torch safetensors to use save_delta_from_paths")
    keys = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".safetensors"):
            with safe_open(f"{folder}/{fname}", framework="pt", device="cpu") as f:
                keys.extend(f.keys())
    return sorted(keys)


def _get_tensor_numpy(folder: str, key: str) -> np.ndarray:
    try:
        import torch
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install safetensors torch to use save_delta_from_paths")
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".safetensors"):
            with safe_open(f"{folder}/{fname}", framework="pt", device="cpu") as f:
                if key in f.keys():
                    return f.get_tensor(key).to(torch.float32).numpy()
    raise KeyError(f"Tensor '{key}' not found in {folder}")


def _get_tensor_numpy_raw(folder: str, key: str) -> np.ndarray:
    try:
        import torch
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install safetensors torch to use save_delta_from_paths")
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".safetensors"):
            with safe_open(f"{folder}/{fname}", framework="pt", device="cpu") as f:
                if key in f.keys():
                    t = f.get_tensor(key)
                    if t.dtype == torch.bfloat16:
                        return t.view(torch.int16).numpy()
                    return t.numpy()
    raise KeyError(f"Tensor '{key}' not found in {folder}")


def _build_key_to_shard(folder: str) -> dict:
    """Map each tensor name to the shard filename it lives in."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install safetensors to use save_delta_from_paths")
    result = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".safetensors"):
            with safe_open(f"{folder}/{fname}", framework="pt", device="cpu") as f:
                for k in f.keys():
                    result[k] = fname
    return result

def _load_base_dir_numpy(folder: str, keys: list[str]) -> tuple[Dict[str, np.ndarray], str]:
    """
    Load requested keys from a safetensors folder, one file open per shard.
    Returns (float32 arrays for math, sha256 hex of raw bytes for verify).
    """
    try:
        import torch
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install safetensors torch to use save_delta_from_paths")

    from collections import defaultdict

    key_to_shard = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".safetensors"):
            with safe_open(f"{folder}/{fname}", framework="pt", device="cpu") as f:
                for k in f.keys():
                    key_to_shard[k] = fname

    shard_to_keys = defaultdict(list)
    for k in keys:
        if k not in key_to_shard:
            raise KeyError(f"Tensor '{k}' not found in {folder}")
        shard_to_keys[key_to_shard[k]].append(k)

    out = {}
    hasher = hashlib.sha256()
    for fname, shard_keys in shard_to_keys.items():
        with safe_open(f"{folder}/{fname}", framework="pt", device="cpu") as f:
            for k in sorted(shard_keys):  # sorted for determinism
                t = f.get_tensor(k)
                # raw bytes for hash (matches save side)
                raw = t.view(torch.int16).numpy() if t.dtype == torch.bfloat16 else t.numpy()
                hasher.update(k.encode("utf-8"))
                hasher.update(raw.tobytes())
                # float32 for math
                if t.dtype == torch.bfloat16:
                    t = t.to(torch.float32)
                out[k] = t.numpy().astype(np.float32)

    return out, hasher.hexdigest()


_DTYPE_MAP_SAFETENSORS = {
    "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "F64": "float64", "I32": "int32", "I64": "int64",
    "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool",
}


def _safetensors_tensor_meta(folder: str) -> dict:
    """Read tensor shapes/dtypes from safetensors headers without loading tensor data."""
    result = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".safetensors"):
            with open(os.path.join(folder, fname), "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len).decode("utf-8"))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                result[name] = {
                    "shape": meta["shape"],
                    "dtype": _DTYPE_MAP_SAFETENSORS.get(meta["dtype"], meta["dtype"].lower()),
                }
    return result


def _tensor_header_entry(strategy: str, shape: list, kwargs: dict) -> dict:
    """
    Build the per-tensor JSON header entry (scalars + _ref placeholders) from shape alone.
    Mirrors what compress() returns, minus the array fields — so the header can be written
    before any tensor data is loaded.
    """
    n = int(np.prod(shape)) if shape else 1
    entry: dict = {"strategy": strategy, "shape": shape, "dtype": "float32"}

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
        entry["n_outliers"] = max(1, int(n * outlier_fraction))

    for field in _ARRAY_FIELDS.get(strategy, []):
        entry[field] = {"_ref": field}

    return entry


def _write_array_to_file(f, name: str, field: str, arr: np.ndarray) -> None:
    """Serialise one numpy array to a file object (no incremental checksum)."""
    tn_enc = name.encode("utf-8")
    fl_enc = field.encode("utf-8")
    dt_enc = str(arr.dtype).encode("utf-8")
    f.write(struct.pack("<I", len(tn_enc))); f.write(tn_enc)
    f.write(struct.pack("<I", len(fl_enc))); f.write(fl_enc)
    f.write(struct.pack("<I", len(dt_enc))); f.write(dt_enc)
    f.write(struct.pack("<I", arr.ndim))
    for dim in arr.shape:
        f.write(struct.pack("<Q", dim))
    data = arr.tobytes()
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


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

    parent_hash = hash_state_dict(base_np)
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
    finetuned_dir = str(finetuned_dir)
    base_dir = str(base_dir)
    _use_gpu = _HAS_GPU and use_gpu

    ft_keys = set(_safetensors_keys(finetuned_dir))
    base_keys = set(_safetensors_keys(base_dir))
    if ft_keys != base_keys:
        only_ft = ft_keys - base_keys
        only_base = base_keys - ft_keys
        msg = "Key mismatch between finetuned and base."
        if only_ft:
            msg += f"\n  Only in finetuned: {sorted(only_ft)}"
        if only_base:
            msg += f"\n  Only in base:      {sorted(only_base)}"
        raise ValueError(msg)

    all_keys = sorted(ft_keys)
    if _use_gpu:
        _device = "GPU"
    else:
        _device = f"CPU ({_GPU_UNAVAILABLE_REASON})" if _GPU_UNAVAILABLE_REASON else "CPU"
    print(f"[deltatensors] streaming {len(all_keys)} tensors (strategy={strategy}, prefetch={prefetch}, device={_device})...")

    # --- pass 1: build header from safetensors metadata (zero tensor I/O) ---
    ft_meta = _safetensors_tensor_meta(finetuned_dir)
    _PLACEHOLDER_HASH = "0" * 64  # SHA-256 hex is always exactly 64 ASCII chars
    header = {
        "parent_hash": _PLACEHOLDER_HASH,
        "strategy": strategy,
        "tensors": {
            name: _tensor_header_entry(strategy, ft_meta[name]["shape"], kwargs)
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
                import torch
                from safetensors import safe_open
                from collections import defaultdict

                base_kts = _build_key_to_shard(base_dir)
                ft_kts = _build_key_to_shard(finetuned_dir)

                base_shard_groups: dict = defaultdict(list)
                for name in all_keys:
                    base_shard_groups[base_kts[name]].append(name)

                for base_shard, skeys in base_shard_groups.items():
                    ft_shard_groups: dict = defaultdict(list)
                    for name in skeys:
                        ft_shard_groups[ft_kts[name]].append(name)

                    # keep base shard open across all its keys but load one tensor at a time
                    with safe_open(f"{base_dir}/{base_shard}", framework="pt", device="cpu") as base_sf:
                        for ft_shard, fkeys in ft_shard_groups.items():
                            with safe_open(f"{finetuned_dir}/{ft_shard}", framework="pt", device="cpu") as ft_sf:
                                for name in sorted(fkeys):
                                    t = base_sf.get_tensor(name)
                                    raw = (t.view(torch.int16).numpy() if t.dtype == torch.bfloat16
                                           else t.numpy()).copy()
                                    base_arr = t.to(torch.float32).numpy()
                                    del t
                                    ft_arr = ft_sf.get_tensor(name).to(torch.float32).numpy()
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
                base_hasher.update(base_raw.tobytes())
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
        f.seek(0)
        content = f.read()
        f.write(hashlib.sha256(content).digest())

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

    with open(path, "rb") as f:
        parent_hash, strategy, compressed_tensors = read_wdelta(f)

    if verify:
        verify_base(base_np, parent_hash)

    reconstructed = {}
    for name, payload in compressed_tensors.items():
        if name not in base_np:
            raise KeyError(f"Tensor '{name}' not found in base model.")
        delta = decompress(payload)
        base_arr = base_np[name].astype(np.float32)
        reconstructed[name] = (base_arr + delta).astype(payload["dtype"])

    print(f"[deltatensors] loaded {Path(path).name}  ({len(reconstructed)} tensors, strategy={strategy})")
    return reconstructed

def load_delta_from_paths(
    path: Union[str, Path],
    base_dir: Union[str, Path],
    verify: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct a fine-tuned model from a .wdelta file and a base model directory.
    Loads each base shard once rather than once per tensor.

    Args:
        path:     Path to the .wdelta file.
        base_dir: Folder containing base model safetensors shards.
        verify:   SHA-256 verify the base before reconstructing (recommended).

    Returns:
        Reconstructed state dict as Dict[str, np.ndarray].
    """
    base_dir = str(base_dir)

    with open(path, "rb") as f:
        parent_hash, strategy, compressed_tensors = read_wdelta(f)

    all_keys = list(compressed_tensors.keys())
    base_arrays, actual_hash = _load_base_dir_numpy(base_dir, all_keys)

    if verify:
        if actual_hash != parent_hash:
            raise ValueError(
                f"Base model hash mismatch.\n"
                f"  Expected : {parent_hash}\n"
                f"  Got      : {actual_hash}\n"
                f"Make sure you're loading the exact base model this delta was computed against."
            )

    reconstructed = {}
    for name, payload in compressed_tensors.items():
        base_arr = base_arrays.pop(name)  # pop to free as we go
        delta = decompress(payload)
        reconstructed[name] = (base_arr + delta).astype(payload["dtype"])
        del base_arr, delta

    print(f"[deltatensors] loaded {Path(path).name}  ({len(reconstructed)} tensors, strategy={strategy})")
    return reconstructed

def inspect(path: Union[str, Path]) -> dict:
    """
    Return metadata from a .wdelta file without loading the base model.
    """
    with open(path, "rb") as f:
        parent_hash, strategy, compressed_tensors = read_wdelta(f)

    size_mb = os.path.getsize(path) / 1e6
    return {
        "path": str(path),
        "size_mb": round(size_mb, 2),
        "parent_hash": parent_hash,
        "strategy": strategy,
        "n_tensors": len(compressed_tensors),
        "tensors": {
            name: {"shape": meta["shape"], "dtype": meta["dtype"]}
            for name, meta in compressed_tensors.items()
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
    verify: bool = True,
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
        verify:      Verify SHA-256 parent_hash at every step (recommended).

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

    if isinstance(base, (str, Path)):
        # streaming first step: avoids loading base fully into RAM
        current: Dict[str, np.ndarray] = load_delta_from_paths(
            delta_paths[0], base, verify=verify
        )
        remaining = delta_paths[1:]
    else:
        current = _to_numpy(base)
        remaining = delta_paths

    for path in remaining:
        current = load_delta(path, current, verify=verify)

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
    finetuned_dir = str(finetuned_dir)
    base_dir = str(base_dir)
    parent_delta_path = str(parent_delta_path)
    out_path = Path(out_path)
    _use_gpu = _HAS_GPU and use_gpu

    print(f"[deltatensors] chain: streaming delta vs {Path(parent_delta_path).name}...")

    # Pass 1: verify parent wdelta checksum (full read, discarded immediately after)
    with open(parent_delta_path, "rb") as _f:
        _raw = _f.read()
    if hashlib.sha256(_raw[:-32]).digest() != _raw[-32:]:
        raise ValueError(f"Parent .wdelta checksum mismatch: {parent_delta_path}")
    del _raw

    # Pass 2: parse header, then stream compressed arrays one tensor at a time
    with open(parent_delta_path, "rb") as pf:
        _magic = pf.read(len(MAGIC))
        if _magic != MAGIC:
            raise ValueError("Parent is not a valid .wdelta file (bad magic bytes)")
        _ver = struct.unpack("<I", pf.read(4))[0]
        if _ver != VERSION:
            raise ValueError(f"Unsupported parent .wdelta version {_ver} (expected {VERSION})")
        _hlen = struct.unpack("<I", pf.read(4))[0]
        parent_header = json.loads(pf.read(_hlen).decode("utf-8"))
        # pf is now positioned at the start of the binary array section;
        # arrays are stored in sorted key order, matching our iteration below

        parent_tensor_metas = parent_header["tensors"]
        parent_strat = parent_header["strategy"]
        parent_fields = _ARRAY_FIELDS.get(parent_strat, [])
        all_keys = sorted(parent_tensor_metas.keys())

        ft_keys = set(_safetensors_keys(finetuned_dir))
        if set(all_keys) != ft_keys:
            only_parent = set(all_keys) - ft_keys
            only_ft = ft_keys - set(all_keys)
            msg = "Key mismatch between parent delta and finetuned model."
            if only_parent:
                msg += f"\n  Only in parent delta: {sorted(only_parent)}"
            if only_ft:
                msg += f"\n  Only in finetuned:    {sorted(only_ft)}"
            raise ValueError(msg)

        print(f"[deltatensors] chain: {len(all_keys)} tensors (strategy={strategy})...")

        # Build output header from finetuned safetensors metadata (zero tensor I/O)
        _PLACEHOLDER_HASH = "0" * 64
        ft_meta = _safetensors_tensor_meta(finetuned_dir)
        out_header = {
            "parent_hash": _PLACEHOLDER_HASH,
            "strategy": strategy,
            "tensors": {
                name: _tensor_header_entry(strategy, ft_meta[name]["shape"], kwargs)
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
                # Read compressed arrays for this tensor from parent wdelta (sequential)
                tensor_fields: dict = {}
                for _ in parent_fields:
                    _tn_len = struct.unpack("<I", pf.read(4))[0]; pf.read(_tn_len)
                    _fl_len = struct.unpack("<I", pf.read(4))[0]
                    _fl = pf.read(_fl_len).decode("utf-8")
                    _dt_len = struct.unpack("<I", pf.read(4))[0]
                    _dtype = np.dtype(pf.read(_dt_len).decode("utf-8"))
                    _ndim = struct.unpack("<I", pf.read(4))[0]
                    _shape = tuple(struct.unpack("<Q", pf.read(8))[0] for _ in range(_ndim))
                    _dlen = struct.unpack("<Q", pf.read(8))[0]
                    tensor_fields[_fl] = np.frombuffer(pf.read(_dlen), dtype=_dtype).reshape(_shape)

                # Reconstruct parent tensor = base_tensor + decompress(parent_delta)
                base_arr = _get_tensor_numpy(base_dir, name)
                parent_arr = (base_arr + decompress({**parent_tensor_metas[name], **tensor_fields})).astype(np.float32)
                del base_arr, tensor_fields

                # Accumulate hash of the reconstructed parent (becomes parent_hash in output)
                parent_hasher.update(name.encode("utf-8"))
                parent_hasher.update(parent_arr.tobytes())

                # Load finetuned tensor and compute delta
                ft_arr = _get_tensor_numpy(finetuned_dir, name)
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
        out_f.seek(0)
        content = out_f.read()
        out_f.write(hashlib.sha256(content).digest())

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[deltatensors] saved {out_path.name}  ({len(all_keys)} tensors, {size_mb:.1f} MB)")
    return parent_hash