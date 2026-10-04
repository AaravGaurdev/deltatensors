"""
Streaming reconstruction: base safetensors + .wdelta -> fine-tuned weights,
one tensor at a time.

Each tensor is reconstructed in a single scratch buffer (WorkBuffer): the raw
base bytes are read into it, hashed if needed, widened to float32 in place,
the delta is added in place, and the result is narrowed in place to the
output dtype. Peak memory is about num_workers x (4 bytes x the largest
tensor's element count), plus whatever the caller keeps.

Tensors are processed by a thread pool (numpy, file reads and hashlib release
the GIL). The base hash is still one sequential SHA-256 over the tensors in a
fixed order: each worker hashes its tensor when its turn comes, and results
are handed to the sink strictly in order, so output is deterministic for any
number of workers.
"""

from __future__ import annotations
import hashlib
import json
import os
import shutil
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np

from . import verify_cache
from .compress import decompress_add_
from .format import WDeltaReader
from .safetensors_io import FLOAT_DTYPES, SafetensorsDir, ShardedWriter, WorkBuffer, dtype_name

PathLike = Union[str, os.PathLike]

VERIFY_MODES = ("full", "cached", "none")

# Files in a model folder that hold weights; everything else at the top level
# (config, tokenizer, generation config, chat template, ...) is copied over.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack", ".gguf", ".wdelta")
_WEIGHT_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def default_num_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _verify_mode(verify) -> str:
    if verify is True:
        return "full"
    if verify is False or verify is None:
        return "none"
    if verify not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {VERIFY_MODES} (or a bool), got {verify!r}")
    return verify


def hash_mismatch_error(expected: str, actual: str, hint: str = "") -> ValueError:
    return ValueError(
        f"Base model hash mismatch.\n"
        f"  Expected : {expected}\n"
        f"  Got      : {actual}\n"
        f"Make sure you're loading the exact base model this delta was computed against.{hint}"
    )


class _Turnstile:
    """Lets tasks 0, 1, 2, ... run a critical section strictly in that order."""

    def __init__(self):
        self._next = 0
        self._failed = False
        self._cond = threading.Condition()

    def wait(self, i: int) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._next == i or self._failed)
            if self._failed:
                raise RuntimeError("aborted: an earlier tensor failed")

    def done(self) -> None:
        with self._cond:
            self._next += 1
            self._cond.notify_all()

    def fail(self) -> None:
        with self._cond:
            self._failed = True
            self._cond.notify_all()


def _check_delta_integrity(reader: WDeltaReader, path: PathLike, mode: str) -> None:
    if mode == "full":
        reader.verify_checksum()
    elif mode == "cached":
        key = verify_cache.make_key("wdelta-checksum", [path])
        if verify_cache.get(key) != "ok":
            reader.verify_checksum()
            verify_cache.put(key, "ok")


def stream_reconstruct(
    wdelta_path: PathLike,
    base_dir: PathLike,
    sink: Callable[[str, np.ndarray, str], None],
    out_dtype: Callable[[str], str],
    verify: Union[str, bool] = "cached",
    num_workers: Optional[int] = None,
    device="cpu",
    include_passthrough: bool = False,
    on_plan: Optional[Callable[[list], None]] = None,
) -> dict:
    """
    Core streaming pass. Calls ``sink(name, array, dtype)`` once per tensor,
    in a deterministic order, where ``array`` is in storage form for ``dtype``
    (bfloat16 as uint16) and may be a view into a scratch buffer that is
    reused once ``sink`` returns.

    ``out_dtype(base_dtype) -> dtype`` picks each tensor's output dtype.
    ``on_plan([(name, dtype, shape), ...])`` is called before any tensor is
    processed. With include_passthrough, base tensors that have no delta are
    emitted too (after the delta'd ones).

    Raises ValueError on a base hash mismatch, after the pass when the hash is
    computed in-pass (callers must discard what the sink received).
    Returns {"parent_hash", "strategy", "version", "verified"}.
    """
    mode = _verify_mode(verify)
    workers = default_num_workers() if num_workers is None else max(1, int(num_workers))
    apply_fn = _device_apply(device)

    with WDeltaReader(wdelta_path) as reader, SafetensorsDir(base_dir) as base:
        _check_delta_integrity(reader, wdelta_path, mode)

        names = reader.names
        missing = [k for k in names if k not in base]
        if missing:
            raise KeyError(f"Tensors not found in base model {base_dir}: {missing[:10]}")
        delta_set = set(names)
        hash_order = sorted(names) if reader.version >= 2 else base.legacy_order(names)
        order = list(hash_order)
        if include_passthrough:
            order += sorted(k for k in base.keys() if k not in delta_set)

        plan = []
        for name in order:
            info = base.tensors[name]
            meta = reader.tensor_metas.get(name)
            if meta is not None and list(meta["shape"]) != list(info.shape):
                raise ValueError(f"Shape mismatch for '{name}': delta {meta['shape']} vs base {list(info.shape)}")
            dst = out_dtype(info.dtype) if (info.dtype in FLOAT_DTYPES) else info.dtype
            plan.append((name, dst, info.shape))
        if on_plan is not None:
            on_plan(plan)

        # --- base verification: decide whether to hash in this pass ---
        need_hash = mode == "full"
        cache_key = None
        hint = ""
        if mode == "cached":
            cache_key = verify_cache.make_key(
                "base-hash", base.shards,
                order="sorted" if reader.version >= 2 else "legacy",
                keys=verify_cache.keys_digest(hash_order),
            )
            cached = verify_cache.get(cache_key)
            if cached is None:
                need_hash = True
            elif cached != reader.parent_hash:
                raise hash_mismatch_error(reader.parent_hash, cached, " (cached result; use verify='full' to re-hash)")
        if reader.version < 2:
            hint = ("\nv1 files written by save_delta() hash in sorted order, which matches "
                    "this base folder only if its shards don't interleave; verify='none' skips the check.")

        hasher = hashlib.sha256() if need_hash else None
        hash_index = {name: i for i, name in enumerate(hash_order)} if need_hash else {}
        turnstile = _Turnstile()

        def work(name: str, dst: str):
            info = base.tensors[name]
            wb = WorkBuffer(info.numel, info.dtype)
            base.read(name, out=wb.raw)
            if name in hash_index:
                turnstile.wait(hash_index[name])
                try:
                    hasher.update(name.encode("utf-8"))
                    hasher.update(memoryview(wb.raw.view(np.uint8)))
                finally:
                    turnstile.done()
            if name in delta_set:
                return apply_fn(reader.payload(name), wb, dst).reshape(info.shape)
            if dst == info.dtype:
                return wb.raw.reshape(info.shape)
            wb.widen()
            return wb.narrow(dst).reshape(info.shape)

        def run_one(item):
            try:
                return work(item[0], item[1])
            except BaseException:
                turnstile.fail()
                raise

        if workers == 1:
            for name, dst, _ in plan:
                sink(name, run_one((name, dst)), dst)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deltatensors") as ex:
                pending: deque = deque()
                it = iter(plan)
                try:
                    for item in it:
                        pending.append((item, ex.submit(run_one, item)))
                        if len(pending) >= workers:
                            break
                    while pending:
                        item, fut = pending.popleft()
                        arr = fut.result()
                        nxt = next(it, None)
                        if nxt is not None:
                            pending.append((nxt, ex.submit(run_one, nxt)))
                        sink(item[0], arr, item[1])
                        del arr
                except BaseException:
                    turnstile.fail()
                    for _, fut in pending:
                        fut.cancel()
                    raise

        verified = False
        if hasher is not None:
            actual = hasher.hexdigest()
            if cache_key is not None:
                verify_cache.put(cache_key, actual)
            if actual != reader.parent_hash:
                raise hash_mismatch_error(reader.parent_hash, actual, hint)
            verified = True
        elif mode == "cached":
            verified = True

        return {"parent_hash": reader.parent_hash, "strategy": reader.strategy,
                "version": reader.version, "verified": verified}


# ---------------------------------------------------------------------------
# Delta application backends
# ---------------------------------------------------------------------------

def _apply_numpy(payload: dict, wb: WorkBuffer, dst: str) -> np.ndarray:
    decompress_add_(payload, wb.widen())
    return wb.narrow(dst)


def _make_torch_apply(device):
    import torch
    from .safetensors_io import storage_dtype
    from .torch_backend import add_delta_torch_, numpy_to_torch, torch_dtype

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")

    def apply(payload: dict, wb: WorkBuffer, dst: str) -> np.ndarray:
        try:
            src = numpy_to_torch(wb.raw, wb.src_dtype)
        except TypeError:  # dtype torch can't wrap (e.g. uint32): numpy path
            return _apply_numpy(payload, wb, dst)
        x = src.to(dev).to(torch.float32)
        add_delta_torch_(payload, x)
        y = x.to(torch_dtype(dst))
        del x
        # copy the result into the head of the scratch buffer (no new host array)
        sd = storage_dtype(dst)
        if sd.itemsize * wb.n <= len(wb.buf):
            out = wb.buf[:sd.itemsize * wb.n].view(sd)
        else:
            out = np.empty(wb.n, dtype=sd)
        numpy_to_torch(out, dst).copy_(y.view(-1))
        return out

    return apply


def _device_apply(device):
    if device is None or str(device) == "cpu":
        return _apply_numpy
    return _make_torch_apply(device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _copy_model_files(base_dir: Path, out_dir: Path, out_dtype: Optional[str]) -> None:
    for src in sorted(base_dir.iterdir()):
        if not src.is_file() or src.name in _WEIGHT_INDEXES or src.suffix in _WEIGHT_SUFFIXES:
            continue
        dst = out_dir / src.name
        if src.name == "config.json" and out_dtype:
            try:
                cfg = json.loads(src.read_text(encoding="utf-8"))
            except ValueError:
                cfg = None
            if isinstance(cfg, dict):
                for key in ("torch_dtype", "dtype"):
                    if key in cfg:
                        cfg[key] = out_dtype
                dst.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                continue
        shutil.copy2(src, dst)


def reconstruct_to_safetensors(
    base_dir: PathLike,
    wdelta_path: PathLike,
    out_dir: PathLike,
    dtype: Optional[str] = "bfloat16",
    shard_size: Union[int, str] = "5GB",
    verify: Union[str, bool] = "cached",
    num_workers: Optional[int] = None,
    device: str = "cpu",
) -> dict:
    """
    Reconstruct a fine-tuned model straight to a safetensors folder, streaming
    one tensor at a time. Peak RAM is about num_workers x the largest tensor
    in float32, independent of model size.

    Args:
        base_dir:    Base model folder (one or more .safetensors shards).
        wdelta_path: The .wdelta file.
        out_dir:     Output folder. Gets model.safetensors (or
                     model-0000i-of-0000N.safetensors + model.safetensors.index.json),
                     plus every non-weight file from base_dir (config, tokenizer, ...).
        dtype:       Output dtype for floating-point tensors: "bfloat16" (default),
                     "float16", "float32", or None to keep each base tensor's dtype.
                     config.json's torch_dtype is updated to match.
        shard_size:  Max bytes per output shard, e.g. "5GB", "500MiB", or an int.
        verify:      "cached" (default): verify the base hash, but skip re-hashing
                     base files whose path, size and mtime match an earlier
                     successful check. "full": always hash. "none": skip.
        num_workers: Threads (default min(8, cpu_count)). Output is identical
                     for any value.
        device:      "cpu" (default) or "cuda" to dequantize and add on the GPU.

    Base tensors without a delta are copied through (cast to ``dtype``).
    Output files are written as *.partial and only renamed into place once
    the base hash has been verified, so a mismatch leaves no partial model.

    Returns a dict with the output files and verification details.
    """
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    if dtype is not None and dtype not in FLOAT_DTYPES:
        raise ValueError(f"dtype must be one of {FLOAT_DTYPES} or None, got {dtype!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    state: dict = {}

    def on_plan(plan):
        state["writer"] = ShardedWriter(out_dir, plan, shard_size=shard_size)

    def sink(name, arr, _dtype):
        state["writer"].write(name, arr)

    try:
        info = stream_reconstruct(
            wdelta_path, base_dir, sink,
            out_dtype=(lambda src: src) if dtype is None else (lambda src: dtype),
            verify=verify, num_workers=num_workers, device=device,
            include_passthrough=True, on_plan=on_plan,
        )
    except BaseException:
        if "writer" in state:
            state["writer"].abort()
        raise

    files = state["writer"].commit()
    _copy_model_files(base_dir, out_dir, dtype)
    print(f"[deltatensors] reconstructed {Path(wdelta_path).name} -> {out_dir} "
          f"({len(files)} shard{'s' if len(files) != 1 else ''}, dtype={dtype or 'base'})")
    return dict(info, files=[str(f) for f in files])


def _numpy_default_dtype(src: str) -> str:
    # numpy has no bfloat16; float32 holds every bfloat16 value exactly
    return "float32" if src == "bfloat16" else src


def load_delta_from_paths(
    path: PathLike,
    base_dir: PathLike,
    verify: Union[str, bool] = "cached",
    dtype: Optional[str] = None,
    num_workers: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, "np.ndarray"]:
    """
    Reconstruct a fine-tuned model from a .wdelta file and a base model folder,
    building the result one tensor at a time (the base is never loaded whole).

    Args:
        path:        Path to the .wdelta file.
        base_dir:    Folder containing the base model's safetensors shards.
        verify:      "cached" (default), "full", or "none"; see
                     reconstruct_to_safetensors. True/False mean "full"/"none".
        dtype:       Output dtype. None (default): each base tensor's own dtype,
                     except bfloat16, which numpy can't represent, becomes
                     float32 (exactly). "bfloat16" returns torch.bfloat16
                     tensors (requires torch). Otherwise a numpy float dtype name.
        num_workers: Threads (default min(8, cpu_count)).
        device:      "cpu" or "cuda" (compute only; results are returned on CPU).

    Returns:
        Dict[str, np.ndarray] (or torch.Tensor for dtype="bfloat16").
    """
    if dtype is not None:
        dtype = dtype_name(dtype)
        if dtype not in FLOAT_DTYPES:
            raise ValueError(f"dtype must be one of {FLOAT_DTYPES} or None, got {dtype!r}")
    if dtype == "bfloat16":
        import torch  # noqa: F401  (fail early, before the pass)

    out: Dict[str, "np.ndarray"] = {}

    def sink(name, arr, dst):
        # keep only the result, not the (possibly larger) scratch buffer it views
        if arr.base is not None and arr.base.nbytes > arr.nbytes:
            arr = arr.copy()
        if dst == "bfloat16":
            import torch
            arr = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
        out[name] = arr

    info = stream_reconstruct(
        path, base_dir, sink,
        out_dtype=_numpy_default_dtype if dtype is None else (lambda src: dtype),
        verify=verify, num_workers=num_workers, device=device,
    )
    print(f"[deltatensors] loaded {Path(path).name}  ({len(out)} tensors, strategy={info['strategy']})")
    return out
