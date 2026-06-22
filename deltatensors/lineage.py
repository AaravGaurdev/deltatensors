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


def hash_state_dict(state_dict: Dict[str, np.ndarray]) -> str:
    """
    Deterministic SHA-256 of a state dict.
    Tensors are hashed in sorted-key order.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        arr = np.asarray(state_dict[key])
        h.update(key.encode("utf-8"))
        h.update(arr.tobytes())
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
