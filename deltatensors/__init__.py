"""
deltatensors — near-lossless delta compression for fine-tuned models.

Quick start:
    import deltatensors as dt

    dt.save_delta("checkpoint.wdelta", finetuned_state_dict, base_state_dict, strategy="sparse")
    reconstructed = dt.load_delta("checkpoint.wdelta", base_state_dict)
    info = dt.inspect("checkpoint.wdelta")

    # streaming, from safetensors folders
    dt.reconstruct_to_safetensors("base-model/", "checkpoint.wdelta", "finetuned-model/")

    # hot-swap fine-tunes in a loaded torch model
    dt.swap_delta_(model, "other.wdelta", base_state)

HuggingFace Trainer integration:
    from deltatensors.training import DeltaTensorsCallback
"""

from .io import (
    save_delta,
    save_delta_from_paths,
    load_delta,
    load_delta_from_paths,
    inspect,
    inspect_chain,
    load_delta_chain,
    save_delta_chain_from_paths,
    reconstruct_to_safetensors,
)
from .lineage import hash_state_dict
from .hotswap import apply_delta_, swap_delta_

__version__ = "0.3.0"
__all__ = [
    "save_delta",
    "save_delta_from_paths",
    "load_delta",
    "load_delta_from_paths",
    "inspect",
    "inspect_chain",
    "load_delta_chain",
    "save_delta_chain_from_paths",
    "reconstruct_to_safetensors",
    "apply_delta_",
    "swap_delta_",
    "hash_state_dict",
]