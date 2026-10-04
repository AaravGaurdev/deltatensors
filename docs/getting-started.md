# Getting Started

## Installation

```bash
pip install deltatensors
pip install torch  # optional: GPU reconstruction, hot-swapping into torch models
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

### Reconstruct to a model folder

```python
dt.reconstruct_to_safetensors(
    "qwen-base/",          # base model directory
    "checkpoint.wdelta",
    "qwen-wiki-rebuilt/",  # output: safetensors shards + index + config/tokenizer
    dtype="bfloat16",      # or "float16", "float32", None (keep the base's dtypes)
)
```

The output folder is a normal Hugging Face checkpoint: `AutoModelForCausalLM.from_pretrained("qwen-wiki-rebuilt/")` loads it. Tensors stream through one at a time, so peak RAM stays around the size of the largest tensor in float32 per worker, regardless of model size.

### Reconstruct into memory

```python
recon_sd = dt.load_delta_from_paths("checkpoint.wdelta", "qwen-base/")
```

Returns a `Dict[str, np.ndarray]` in the base's dtypes (bfloat16 comes back as float32, since numpy has no bfloat16; pass `dtype="bfloat16"` to get torch bfloat16 tensors instead). The base is never loaded whole, but the result is a full model in RAM.

Both functions verify the base model's SHA-256 against the `parent_hash` stored in the `.wdelta`, because applying a delta to the wrong base produces garbage silently. With the default `verify="cached"`, the hash is computed during the reconstruction pass and remembered (keyed on each base shard's path, size and mtime), so reconstructing more fine-tunes of the same base skips it. `verify="full"` hashes every time; `verify="none"` skips the check.

### Apply to a loaded model (and hot-swap)

To run a fine-tune without writing it out, load the base once and add the delta in place:

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("qwen-base/", torch_dtype=torch.bfloat16).cuda()
dt.apply_delta_(model, "checkpoint.wdelta")
```

This works tensor by tensor on the parameters' device, so the extra memory is one float32 tensor at a time. To switch between fine-tunes, keep a copy of the base weights and swap:

```python
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  # GPU or CPU

dt.apply_delta_(model, "math.wdelta")
dt.swap_delta_(model, "code.wdelta", base_state)
```

`swap_delta_` restores every touched weight from `base_state` before adding the new delta. Don't undo a delta by subtracting it: bf16 addition isn't reversible, and the error accumulates with every swap.

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
#     'model.embed_tokens.weight': {'shape': [151936, 896], 'dtype': 'bfloat16'},
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

`save_delta_chain_from_paths` is streaming — it reads the parent `.wdelta` one tensor at a time without ever reconstructing the full parent model in RAM.

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
    verify="cached",   # verifies parent_hash at each step
)
```

The result is float32: chain hashes are defined over float32 models, so intermediate steps stay float32. Applying deltas in the wrong order raises `ValueError: hash mismatch` immediately. Pass a directory path for `base` (uses the streaming loader for the first step) or an in-memory state dict.

### Flat vs chained

- **Flat**: all deltas computed against the original base. Each delta can be applied independently; reconstruction always requires only the base + one wdelta.
- **Chained**: each delta computed against the prior model. Smaller delta magnitudes → better reconstruction quality at the same compression ratio. Reconstruction requires the full chain from the beginning.

Use flat when checkpoints are independent experiments; use chained when you're tracking a sequential training trajectory (continual learning, multi-stage RLHF, iterative fine-tuning).
