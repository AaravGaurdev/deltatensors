# Getting Started

## Installation

```bash
pip install deltatensors
pip install torch safetensors  # for loading from safetensors directories
```

Requires Python 3.9+.

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

---

## HuggingFace Trainer integration

`DeltaTensorsCallback` hooks into the HuggingFace `Trainer` and saves each checkpoint as a `.wdelta` file automatically. The GPU is always free during the save (the callback forces CPU compression to avoid competing with optimizer states).

```python
from deltatensors.training import DeltaTensorsCallback
from transformers import Trainer, TrainingArguments

callback = DeltaTensorsCallback(
    base_dir="path/to/base-model",   # the model training started from
    strategy="int4",
    outlier_fraction=0.05,
    delete_full_checkpoint=False,     # True: saves disk, but can't resume or use load_best_model_at_end
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="outputs",
        save_steps=500,
        ...
    ),
    callbacks=[callback],
    ...
)
trainer.train()
```

After training, each checkpoint directory contains `model.wdelta` alongside (or instead of, if `delete_full_checkpoint=True`) the safetensors files:

```
outputs/
  checkpoint-500/
    model.wdelta        ← delta vs base_dir
    model.safetensors   ← kept unless delete_full_checkpoint=True
  checkpoint-1000/
    model.wdelta
    model.safetensors
```

Reconstruct any checkpoint:

```python
sd = dt.load_delta_from_paths(
    "outputs/checkpoint-500/model.wdelta",
    "path/to/base-model",
)
```

**`delete_full_checkpoint=True` warning:** Removing the safetensors files saves disk but prevents resuming training from that checkpoint and prevents `load_best_model_at_end` from working. Only use it for checkpoints you won't resume from.

---

## Lineage chains

Chains let you track a full fine-tuning history. Instead of each delta being against the original base, each delta is against the *prior reconstructed model* — so incremental updates stay small in magnitude:

```
base ──► v1.wdelta ──► v1_model ──► v2.wdelta ──► v2_model
```

The `parent_hash` field in each `.wdelta` file is the SHA-256 of the model it was computed against, forming a verifiable chain.

### Save a chained delta

```python
# v1: normal delta vs base
dt.save_delta_from_paths("v1.wdelta", "v1_checkpoint/", "base_model/", strategy="int4")

# v2: chained delta vs reconstructed v1 (not vs base)
dt.save_delta_chain_from_paths(
    "v2.wdelta",
    finetuned_dir="v2_checkpoint/",
    parent_delta_path="v1.wdelta",
    base_dir="base_model/",
    strategy="int4",
    outlier_fraction=0.05,
)
```

`save_delta_chain_from_paths` is fully streaming — it reads the parent `.wdelta` one tensor at a time without ever reconstructing the full parent model in RAM.

### Inspect chain metadata

```python
history = dt.inspect_chain(["v1.wdelta", "v2.wdelta", "v3.wdelta"])
for entry in history:
    print(f"step {entry['step']}: {entry['size_mb']:.1f} MB  parent={entry['parent_hash'][:8]}")
```

Returns a list of dicts with the same fields as `inspect()` plus a `step` index. No tensors are loaded.

### Reconstruct the final model

```python
# Apply the full chain from base → v1 → v2
sd = dt.load_delta_chain(
    ["v1.wdelta", "v2.wdelta"],
    base="base_model/",
    verify=True,   # verifies parent_hash at each step
)
```

With `verify=True`, applying deltas in the wrong order raises `ValueError: hash mismatch` immediately. Pass a directory path for `base` (uses the streaming loader for the first step) or an in-memory state dict.

### Flat vs chained

- **Flat**: all deltas computed against the original base. Each delta can be applied independently; reconstruction always requires only the base + one wdelta.
- **Chained**: each delta computed against the prior model. Smaller delta magnitudes → better reconstruction quality at the same compression ratio. Reconstruction requires the full chain from the beginning.

Use flat when checkpoints are independent experiments; use chained when you're tracking a sequential training trajectory (continual learning, multi-stage RLHF, iterative fine-tuning).
