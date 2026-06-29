"""
deltatensors — lossless delta-compressed weight format for fine-tuned models.

Quick start:
    import deltatensors as dt

    dt.save_delta("checkpoint.wdelta", finetuned_state_dict, base_state_dict, strategy="sparse")
    reconstructed = dt.load_delta("checkpoint.wdelta", base_state_dict)
    info = dt.inspect("checkpoint.wdelta")

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
)
from .lineage import hash_state_dict

__version__ = "0.2.0"
__all__ = [
    "save_delta",
    "save_delta_from_paths",
    "load_delta",
    "load_delta_from_paths",
    "inspect",
    "inspect_chain",
    "load_delta_chain",
    "save_delta_chain_from_paths",
    "hash_state_dict",
]