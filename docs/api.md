# API Reference

## save_delta_from_paths

```python
dt.save_delta_from_paths(
    out_path,
    finetuned_dir,
    base_dir,
    strategy="sparse",
    prefetch=2,
    use_gpu=True,
    **kwargs,
) -> str
```

Streaming delta save. Peak RAM is O(prefetch tensors), not O(two full models).

A producer thread reads tensor pairs from disk; a writer thread drains compressed output to disk concurrently. The header is written first (from safetensors metadata, no tensor I/O), and the `parent_hash` is seek-patched after streaming completes.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `out_path` | `str \| Path` | Output `.wdelta` file path |
| `finetuned_dir` | `str \| Path` | Folder containing fine-tuned safetensors shards |
| `base_dir` | `str \| Path` | Folder containing base safetensors shards |
| `strategy` | `str` | `"sparse"`, `"quantized"`, or `"int4"` |
| `prefetch` | `int` | Read/write queue depth (default 2) |
| `use_gpu` | `bool` | Use CuPy GPU compression if available (default `True`) |
| `**kwargs` | | Strategy-specific options (see below) |

**Strategy kwargs:**

| Strategy | kwarg | default | description |
|---|---|---|---|
| `sparse` | `sparsity` | `0.9` | Fraction of weights to zero out |
| `int4` | `outlier_fraction` | `0.01` | Fraction of weights stored as float16 outliers |

**Returns:** SHA-256 hex hash of the base model.

---

## load_delta_from_paths

```python
dt.load_delta_from_paths(
    path,
    base_dir,
    verify=True,
) -> Dict[str, np.ndarray]
```

Reconstruct a fine-tuned model from a `.wdelta` file and a base model directory. Loads each base shard once — O(n_shards) file opens rather than O(n_tensors × n_shards).

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.wdelta` file |
| `base_dir` | `str \| Path` | Folder containing base safetensors shards |
| `verify` | `bool` | SHA-256 verify base before reconstructing (default `True`) |

**Returns:** Reconstructed state dict as `Dict[str, np.ndarray]`.

---

## inspect

```python
dt.inspect(path) -> dict
```

Return metadata from a `.wdelta` file without loading the base model.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.wdelta` file |

**Returns:**
```python
{
    "path": "checkpoint.wdelta",
    "size_mb": 294.2,
    "parent_hash": "e1810a...",  # SHA-256 of the base model
    "strategy": "int4",
    "n_tensors": 290,
    "tensors": {
        "model.embed_tokens.weight": {"shape": [151936, 896], "dtype": "float32"},
        ...
    }
}
```

---

## inspect_chain

```python
dt.inspect_chain(delta_paths) -> list
```

Return metadata for each step in a lineage chain without loading any tensors.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `delta_paths` | `list[str \| Path]` | Ordered list of `.wdelta` paths, oldest first |

**Returns:** List of dicts, one per file. Each dict has the same fields as `inspect()` plus a `"step"` key (0-indexed).

```python
history = dt.inspect_chain(["v1.wdelta", "v2.wdelta", "v3.wdelta"])
for entry in history:
    print(entry["step"], entry["size_mb"], "MB", entry["parent_hash"][:8])
```

The `parent_hash` of step N should equal `hash_state_dict(model_produced_by_step_N-1)` for a valid chain. This is verified automatically when loading via `load_delta_chain(..., verify=True)`.

---

## load_delta_chain

```python
dt.load_delta_chain(
    delta_paths,
    base,
    verify=True,
) -> Dict[str, np.ndarray]
```

Reconstruct the final model by applying a sequence of delta files in order.

```
base ──► delta_paths[0] ──► model_1 ──► delta_paths[1] ──► model_2 ──► …
```

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `delta_paths` | `list[str \| Path]` | Ordered list of `.wdelta` paths, oldest first |
| `base` | `str \| Path \| Dict` | Base safetensors directory **or** in-memory state dict |
| `verify` | `bool` | Verify `parent_hash` at each step (default `True`) |

**Returns:** Reconstructed state dict at the end of the chain.

With `verify=True`, applying deltas in the wrong order raises `ValueError: hash mismatch` immediately. Passing a directory for `base` uses the streaming loader for the first step.

---

## save_delta_chain_from_paths

```python
dt.save_delta_chain_from_paths(
    out_path,
    finetuned_dir,
    parent_delta_path,
    base_dir,
    strategy="sparse",
    use_gpu=True,
    **kwargs,
) -> str
```

Save a chained delta — the difference between `finetuned_dir` and the model reconstructed from `parent_delta_path`.

Fully streaming: the parent `.wdelta` is read one tensor at a time (in sorted key order, matching the on-disk layout). Base and finetuned tensors are loaded on demand and freed immediately. Peak RAM is O(one tensor pair), not O(two full models).

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `out_path` | `str \| Path` | Output `.wdelta` file |
| `finetuned_dir` | `str \| Path` | Folder containing new finetuned safetensors |
| `parent_delta_path` | `str \| Path` | The immediately prior `.wdelta` in the chain |
| `base_dir` | `str \| Path` | Original base model safetensors (needed to reconstruct parent tensor-by-tensor) |
| `strategy` | `str` | `"sparse"`, `"quantized"`, or `"int4"` |
| `use_gpu` | `bool` | Use CuPy GPU compression if available (default `True`) |
| `**kwargs` | | Strategy-specific options |

**Returns:** SHA-256 hash of the reconstructed parent model (stored as `parent_hash` in the output file).

---

## save_delta

```python
dt.save_delta(
    path,
    finetuned,
    base,
    strategy="sparse",
    use_gpu=True,
    **kwargs,
) -> str
```

Compute and save the delta between `finetuned` and `base`. Loads both models fully into RAM — for models larger than ~3B use `save_delta_from_paths` instead.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Output `.wdelta` file path |
| `finetuned` | `Dict[str, np.ndarray \| Tensor]` | Fine-tuned state dict |
| `base` | `Dict[str, np.ndarray \| Tensor]` | Base state dict |
| `strategy` | `str` | `"sparse"`, `"quantized"`, or `"int4"` |
| `use_gpu` | `bool` | Use CuPy GPU compression if available (default `True`) |

**Returns:** SHA-256 hex hash of the base model.

---

## load_delta

```python
dt.load_delta(
    path,
    base,
    verify=True,
) -> Dict[str, np.ndarray]
```

Reconstruct a fine-tuned model from a `.wdelta` file and a base state dict. Requires the full base loaded in RAM — for large models use `load_delta_from_paths` instead.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.wdelta` file |
| `base` | `Dict[str, np.ndarray \| Tensor]` | Base state dict |
| `verify` | `bool` | SHA-256 verify base before reconstructing (default `True`) |

**Returns:** Reconstructed state dict as `Dict[str, np.ndarray]`.

---

## hash_state_dict

```python
dt.hash_state_dict(state_dict) -> str
```

Compute the SHA-256 hash of a state dict. The hash is computed over tensor names and raw bytes in sorted key order — the same hash stored as `parent_hash` in every `.wdelta` file.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `state_dict` | `Dict[str, np.ndarray]` | State dict to hash |

**Returns:** 64-character SHA-256 hex string.

Useful for verifying a chain link manually:

```python
v1_sd = dt.load_delta_from_paths("v1.wdelta", "base/")
assert dt.hash_state_dict(v1_sd) == dt.inspect("v2_chained.wdelta")["parent_hash"]
```

---

## DeltaTensorsCallback

```python
from deltatensors.training import DeltaTensorsCallback

DeltaTensorsCallback(
    base_dir,
    strategy="int4",
    delete_full_checkpoint=False,
    **strategy_kwargs,
)
```

HuggingFace `Trainer` callback that saves each checkpoint as a `.wdelta` file against a fixed base model.

Called automatically by the Trainer after every checkpoint save. The delta is written to `{checkpoint_dir}/model.wdelta`. GPU compression is forced off during training (the GPU is occupied by model weights, optimizer states, and activations); use `save_delta_from_paths` with `use_gpu=True` for standalone post-training compression.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `base_dir` | `str \| Path` | Path to the base model safetensors directory |
| `strategy` | `str` | Compression strategy: `"sparse"`, `"quantized"`, or `"int4"` (default) |
| `delete_full_checkpoint` | `bool` | If `True`, remove `.safetensors` files after saving the delta. Prevents resuming training and `load_best_model_at_end`. Default `False`. |
| `**strategy_kwargs` | | Forwarded to the compression strategy, e.g. `outlier_fraction=0.05` or `sparsity=0.9` |

**Example:**

```python
from deltatensors.training import DeltaTensorsCallback
from transformers import Trainer, TrainingArguments

callback = DeltaTensorsCallback(
    base_dir="path/to/base-model",
    strategy="int4",
    outlier_fraction=0.05,
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="outputs", save_steps=500, ...),
    callbacks=[callback],
)
trainer.train()

# Reconstruct any checkpoint:
sd = dt.load_delta_from_paths("outputs/checkpoint-500/model.wdelta", "path/to/base-model")
```
