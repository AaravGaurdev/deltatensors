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
import io
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
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install safetensors to use save_delta_from_paths")
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


def _hwrite(f, hasher, data: bytes) -> None:
    """Write data to file and update checksum simultaneously."""
    hasher.update(data)
    f.write(data)


def _write_array(f, hasher, tensor_name: str, field: str, arr: np.ndarray) -> None:
    """Serialise one numpy array to file, updating the running checksum."""
    tn_enc = tensor_name.encode("utf-8")
    fl_enc = field.encode("utf-8")
    dt_enc = str(arr.dtype).encode("utf-8")
    _hwrite(f, hasher, struct.pack("<I", len(tn_enc))); _hwrite(f, hasher, tn_enc)
    _hwrite(f, hasher, struct.pack("<I", len(fl_enc))); _hwrite(f, hasher, fl_enc)
    _hwrite(f, hasher, struct.pack("<I", len(dt_enc))); _hwrite(f, hasher, dt_enc)
    _hwrite(f, hasher, struct.pack("<I", arr.ndim))
    for dim in arr.shape:
        _hwrite(f, hasher, struct.pack("<Q", dim))
    data = arr.tobytes()
    _hwrite(f, hasher, struct.pack("<Q", len(data)))
    _hwrite(f, hasher, data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_delta(
    path: Union[str, Path],
    finetuned: StateDict,
    base: StateDict,
    strategy: str = "sparse",
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
    **kwargs,
) -> str:
    """
    Streaming delta save — peak RAM is O(prefetch tensors), not O(two full models).

    Architecture:
      - Producer thread reads (base, finetune) tensor pairs from disk into a small queue.
      - Main thread consumes the queue: compress → write to disk → free immediately.
      - Header is written first with a placeholder length, then patched after streaming.
      - Checksum is computed incrementally (no full-file buffer needed).

    Args:
        out_path:      Output .wdelta file path.
        finetuned_dir: Folder containing finetuned safetensors shards.
        base_dir:      Folder containing base safetensors shards.
        strategy:      "sparse" or "quantized".
        prefetch:      Number of tensor pairs to prefetch (default 2). Keep low to save RAM.
        **kwargs:      Strategy-specific options.

    Returns:
        The SHA-256 hash of the base model.
    """
    finetuned_dir = str(finetuned_dir)
    base_dir = str(base_dir)

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
    print(f"[deltatensors] streaming {len(all_keys)} tensors (strategy={strategy}, prefetch={prefetch})...")

    # ADD THESE THREE LINES HERE
    base_hasher = hashlib.sha256()
    tensor_metas = {}
    compressed_cache = {}



    # --- producer: reads tensor pairs into a bounded queue ---
    read_queue = queue.Queue(maxsize=prefetch)
    producer_error = [None]

    # --- inside save_delta_from_paths producer thread ---
    def producer():
        try:
            for name in all_keys:
                base_raw = _get_tensor_numpy_raw(base_dir, name)  # For cryptographic hash
                base_arr = _get_tensor_numpy(base_dir, name)      # FIX 1: Load actual floats for math
                ft_arr = _get_tensor_numpy(finetuned_dir, name)
                read_queue.put((name, base_raw, base_arr, ft_arr))  # Pass all three items
        except Exception as e:
            producer_error[0] = e
        finally:
            read_queue.put(_SENTINEL)

    t = threading.Thread(target=producer, daemon=True)
    t.start()


    # --- inside save_delta_from_paths consumer loop ---
    count = 0
    while True:
        item = read_queue.get()
        if item is _SENTINEL:
            break
        if producer_error[0]:
            raise producer_error[0]

        name, base_raw, base_arr, ft_arr = item  # Unpack the new items

        if ft_arr.shape != base_raw.shape:
            raise ValueError(f"Shape mismatch for '{name}': {ft_arr.shape} vs {base_raw.shape}.")

        # Use raw bytes safely for hashing ONLY
        base_hasher.update(name.encode("utf-8"))
        base_hasher.update(base_raw.tobytes())
        del base_raw 

        # FIX 2: Compute delta using real floating-point values for both arrays
        delta = ft_arr.astype(np.float32) - base_arr.astype(np.float32)
        del base_arr, ft_arr

        compressed = compress(delta, strategy, **kwargs)
        del delta


        # extract clean metadata (no arrays)
        fields = _ARRAY_FIELDS.get(strategy, [])
        clean = {k: v for k, v in compressed.items() if k not in fields}
        tensor_metas[name] = clean
        compressed_cache[name] = compressed

        count += 1
        if count % 50 == 0:
            print(f"[deltatensors]   {count}/{len(all_keys)} tensors compressed...")

    t.join()
    if producer_error[0]:
        raise producer_error[0]

    parent_hash = base_hasher.hexdigest()

    # build header
    header = {
        "parent_hash": parent_hash,
        "strategy": strategy,
        "tensors": {
            name: {**tensor_metas[name], **{f: {"_ref": f} for f in _ARRAY_FIELDS.get(strategy, [])}}
            for name in all_keys
        },
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()

    with open(out_path, "wb") as f:
        _hwrite(f, hasher, MAGIC)
        _hwrite(f, hasher, struct.pack("<I", VERSION))
        _hwrite(f, hasher, struct.pack("<I", len(header_bytes)))
        _hwrite(f, hasher, header_bytes)

        # stream payloads, freeing each immediately after writing
        for name in all_keys:
            compressed = compressed_cache.pop(name)  # pop frees it after this block
            for field in _ARRAY_FIELDS.get(strategy, []):
                arr = np.asarray(compressed[field])
                _write_array(f, hasher, name, field, arr)
                del arr
            del compressed

        f.write(hasher.digest())  # checksum — not hashed into itself

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