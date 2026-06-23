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
    strategy="int4",
    outlier_fraction=0.01,
)
```

This streams tensor pairs from disk one at a time — peak RAM is O(1 tensor), not O(two full models). For models that fit comfortably in RAM, see [in-memory usage](#in-memory-usage-small-models).

### Reconstruct

```python
recon_sd = dt.load_delta_from_paths(
    "checkpoint.wdelta",
    "qwen-base/",
    verify=True,
)
```

Returns a `Dict[str, np.ndarray]`. `verify=True` checks the base model's SHA-256 hash against the one stored in the `.wdelta` file — recommended, since applying a delta to the wrong base produces garbage silently.

### Load into a HuggingFace model

`load_delta_from_paths` gives you a numpy state dict. To run inference you need to patch it into a model. The trick is to do it in-place so you don't hold a full second copy in RAM:

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

Peak RAM here is one loaded model + one delta tensor at a time.

### Inspect without loading anything

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

Useful for checking what base model a `.wdelta` was built against (`parent_hash`) before you bother loading anything.

## Choosing a strategy

`int4` is the default recommendation — it gave 0.58% perplexity difference at 3.2x compression on Qwen2.5-0.5B. Use `sparse` if you want to tune the quality/compression tradeoff manually via `sparsity=`. `quantized` is the most aggressive and will show more quality loss.

| Strategy | Use when |
|---|---|
| `int4` | Best compression with near-lossless quality |
| `sparse` | Tunable tradeoff via `sparsity=0.0` to `0.99` |
| `quantized` | Maximum compression, more quality loss |

## In-memory usage (small models)

If your models fit in RAM you can skip the path-based API and pass state dicts directly:

```python
finetuned_sd = {...}  # Dict[str, np.ndarray] or Dict[str, torch.Tensor]
base_sd = {...}

dt.save_delta("checkpoint.wdelta", finetuned_sd, base_sd, strategy="int4")
recon_sd = dt.load_delta("checkpoint.wdelta", base_sd, verify=True)
```