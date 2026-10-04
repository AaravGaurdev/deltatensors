"""
Lineage utilities: SHA-256 fingerprinting of base models.

The hash is computed over the raw bytes of all tensor values,
sorted by tensor name for determinism.
"""

from __future__ import annotations
import hashlib
import json
from typing import Dict
import numpy as np


# TODO(format v3): switch to BLAKE3 with a per-tensor hash tree, so hashing can
# run in parallel across tensors and threads. SHA-256 over one ordered stream is
# kept here because changing it would change every parent_hash.


def _storage_array(value) -> np.ndarray:
    """numpy view of a tensor's stored bytes (torch bfloat16 as int16 bits)."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        t = value.detach().cpu()
        if str(t.dtype) == "torch.bfloat16":
            import torch
            t = t.view(torch.int16)
        return t.numpy()
    return np.asarray(value)


def hash_state_dict(state_dict: Dict[str, np.ndarray]) -> str:
    """
    Deterministic SHA-256 of a state dict: name and raw bytes of each tensor,
    in sorted-key order. Accepts numpy arrays or torch tensors (bfloat16 is
    hashed as its raw bits, matching what is stored in safetensors).
    """
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        arr = np.ascontiguousarray(_storage_array(state_dict[key]))
        h.update(key.encode("utf-8"))
        h.update(memoryview(arr.reshape(-1).view(np.uint8)))
    return h.hexdigest()


def verify_base(state_dict: Dict[str, np.ndarray], expected_hash: str) -> None:
    """
    Raise if the base model doesn't match the hash stored in the .wdelta file.
    This is the core safety guarantee: you can't accidentally reconstruct
    a model from the wrong base.
    """
    actual = hash_state_dict(state_dict)
    if actual != expected_hash:
        raise ValueError(
            f"Base model hash mismatch.\n"
            f"  Expected : {expected_hash}\n"
            f"  Got      : {actual}\n"
            f"Make sure you're loading the exact base model this delta was computed against."
        )
