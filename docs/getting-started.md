# Getting Started

## Installation

```bash
pip install deltatensors
pip install torch safetensors  # for loading from safetensors directories
```

## Basic usage

### Save a delta

```python
import deltatensors as dt

dt.save_delta_from_paths(
    "checkpoint.wdelta",
    "qwen-wiki/",       # fine-tuned model directory
    "qwen-base/",       # base model directory
    strategy="int4",    # compression strategy
    outlier_fraction=0.01,
)
```

This streams tensor pairs from disk one at a time. Peak RAM is O(1 tensor), not O(two full models).

### Reconstruct

```python
recon_sd = dt.load_delta_from_paths(
    "checkpoint.wdelta",
    "qwen-base/",
    verify=True,    # SHA-256 verify base before reconstructing (recommended)
)
```

Returns a `Dict[str, np.ndarray]`, which is the reconstructed state dict.

### Load into a HuggingFace model

```python
from transformers import AutoModelForCausalLM
from deltatensors.format import read_wdelta
from deltatensors.compress import decompress
import torch

model = AutoModelForCausalLM.from_pretrained("qwen-base/", dtype=torch.float32)
sd = model.state_dict()

with open("checkpoint.wdelta", "rb") as f:
    _, _, compressed_tensors = read_wdelta(f)

for name, payload in compressed_tensors.items():
    if name not in sd:
        continue
    delta = torch.from_numpy(decompress(payload))
    sd[name].add_(delta.to(sd[name].dtype))
    del delta

model.load_state_dict(sd, strict=False)
```

This patches the base model in-place. Peak RAM is one model + one delta tensor at a time.

### Inspect a delta file

```python
info = dt.inspect("checkpoint.wdelta")
print(info)
# {
#   'path': 'checkpoint.wdelta',
#   'size_mb': 294.2,
#   'parent_hash': 'e1810a...',
#   'strategy': 'int4',
#   'n_tensors': 290,
#   'tensors': {
#     'model.embed_tokens.weight': {'shape': [151936, 896], 'dtype': 'float32'},
#     ...
#   }
# }
```

## Choosing a strategy

| Strategy | Use when |
|---|---|
| `int4` | You want best compression with near-lossless quality |
| `sparse` | You want a tunable quality/compression tradeoff |
| `quantized` | You want maximum compression and can tolerate more loss |

## In-memory usage (small models)

If your models fit in RAM, you can pass state dicts directly:

```python
import numpy as np

finetuned_sd = {...}  # Dict[str, np.ndarray] or Dict[str, torch.Tensor]
base_sd = {...}

dt.save_delta("checkpoint.wdelta", finetuned_sd, base_sd, strategy="int4")
recon_sd = dt.load_delta("checkpoint.wdelta", base_sd, verify=True)
```