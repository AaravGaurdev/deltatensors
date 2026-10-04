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

## reconstruct_to_safetensors

```python
dt.reconstruct_to_safetensors(
    base_dir,
    wdelta_path,
    out_dir,
    dtype="bfloat16",
    shard_size="5GB",
    verify="cached",
    num_workers=None,
    device="cpu",
) -> dict
```

Reconstruct a fine-tuned model straight to a safetensors folder, one tensor at a time. The output loads like any Hugging Face checkpoint (`AutoModel.from_pretrained(out_dir)`).

For each tensor, the raw base bytes are read into a single scratch buffer, hashed (when verifying), widened to float32 in place, the delta is added in place, and the result is narrowed in place to `dtype` and written to the current output shard. Peak RAM is about `num_workers` × (4 bytes × the largest tensor's element count) plus the interpreter, independent of model size; see `benchmarks/results.md` for measurements.

Non-weight files in `base_dir` (config, tokenizer, generation config, ...) are copied, and `config.json`'s `torch_dtype` is set to `dtype`. Base tensors that have no delta are copied through (cast to `dtype`; integer tensors keep their dtype). Shards are written as `*.partial` and only renamed into place after the base hash checks out, so a mismatch leaves no half-written model.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `base_dir` | `str \| Path` | Base model folder (one or more `.safetensors` shards) |
| `wdelta_path` | `str \| Path` | The `.wdelta` file |
| `out_dir` | `str \| Path` | Output folder: `model.safetensors`, or `model-0000i-of-0000N.safetensors` + `model.safetensors.index.json` |
| `dtype` | `str \| None` | `"bfloat16"` (default), `"float16"`, `"float32"`, or `None` to keep each base tensor's dtype |
| `shard_size` | `int \| str` | Max bytes per output shard, e.g. `"5GB"` (decimal), `"500MiB"` (binary), or an int |
| `verify` | `str \| bool` | See [verify modes](#verify-modes) |
| `num_workers` | `int \| None` | Threads across tensors (default `min(8, os.cpu_count())`). Output is byte-identical for any value |
| `device` | `str` | `"cpu"` (default, numpy) or `"cuda"` (torch on the GPU, bit-identical results) |

**Returns:** `{"parent_hash", "strategy", "version", "verified", "files"}`.

bfloat16 rounding is round-to-nearest-even, matching `torch.Tensor.to(torch.bfloat16)`. torch is not needed except for `device="cuda"`.

---

## load_delta_from_paths

```python
dt.load_delta_from_paths(
    path,
    base_dir,
    verify="cached",
    dtype=None,
    num_workers=None,
    device="cpu",
) -> Dict[str, np.ndarray]
```

Reconstruct a fine-tuned model from a `.wdelta` file and a base model directory into memory. The base is never loaded whole: the result is built tensor by tensor with the same engine as `reconstruct_to_safetensors`, so peak RAM is the returned dict plus about `num_workers` scratch tensors.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.wdelta` file |
| `base_dir` | `str \| Path` | Folder containing base safetensors shards |
| `verify` | `str \| bool` | See [verify modes](#verify-modes) (default `"cached"`) |
| `dtype` | `str \| None` | `None` (default): each base tensor's own dtype, except bfloat16, which numpy can't represent and becomes float32 (every bfloat16 value is exact in float32). `"bfloat16"` returns `torch.bfloat16` tensors (needs torch). Or `"float16"`, `"float32"`, `"float64"` |
| `num_workers` | `int \| None` | Threads (default `min(8, os.cpu_count())`) |
| `device` | `str` | `"cpu"` or `"cuda"` (compute only; results come back on the CPU) |

**Returns:** `Dict[str, np.ndarray]` (or `torch.Tensor` values for `dtype="bfloat16"`).

0.2.0 returned float32 for everything, after loading the whole base as float32 first.

---

## Verify modes

Every reconstruction checks that the base model is the one the delta was computed against: the SHA-256 of the base tensors must equal the `parent_hash` stored in the `.wdelta`. The hash is computed during the same streaming pass, from bytes that are already in memory, so it costs no extra reads.

| `verify` | Behaviour |
|---|---|
| `"cached"` (default) | Verify, but remember each successful result in `~/.cache/deltatensors` (override with `DELTATENSORS_CACHE_DIR`), keyed on the absolute path, size and modification time of every base shard. Later runs against unchanged files skip hashing; editing, replacing or touching any shard invalidates the entry. The `.wdelta` checksum is cached the same way. |
| `"full"` | Hash every time (0.2.0 behaviour). Use this if files can change without their size or mtime changing. |
| `"none"` | Skip both the base hash and the `.wdelta` checksum. |

`True` and `False` are accepted and mean `"full"` and `"none"`. A mismatch raises `ValueError: Base model hash mismatch`.

---

## apply_delta_

```python
dt.apply_delta_(model, wdelta_path, base_state=None, verify="cached") -> list
```

Add a `.wdelta` into a loaded torch model in place, tensor by tensor, with `param.add_(delta)` on whatever device each parameter lives on. Parameter names must match the delta's tensor names (they do for Hugging Face models loaded from the base checkpoint).

- Without `base_state`, the model must currently hold exactly the base weights; the hash check enforces this, so applying a delta twice raises instead of double-adding.
- With `base_state` (a dict of base tensors on CPU or GPU), every parameter the delta touches is first restored from it.

Keys, shapes and the base hash are all checked before any parameter is modified. Each updated parameter equals `round(float32(base) + delta)` in the parameter's dtype, bit-identical to `reconstruct_to_safetensors`. Tied weights (e.g. `lm_head.weight` sharing `embed_tokens.weight`) are updated once. With `verify="cached"`, a set of base tensors is hashed once per process and re-hashed only if one of them is modified in place (torch's tensor version counter) or replaced; writes through `tensor.data` bypass that counter, so use `verify="full"` if you do those.

**Returns:** the names of the tensors updated.

---

## swap_delta_

```python
dt.swap_delta_(model, new_wdelta_path, base_state, verify="cached") -> list
```

Replace the delta a model currently carries with another one. Every parameter touched by the previously applied delta (as recorded by `apply_delta_`/`swap_delta_` on this model, or every key in `base_state` if unknown) or by the new delta is copied back from `base_state`, then the new delta is added. `base_state` is never modified.

**Never undo a delta by subtracting it.** bfloat16 and float16 addition is not invertible: `round(round(w + d) - d)` often isn't `w`, so subtract-then-add drifts further from the base with every swap. Restoring from `base_state` makes the result after any number of swaps identical to a fresh reconstruction.

```python
model = AutoModelForCausalLM.from_pretrained("base/", torch_dtype=torch.bfloat16).cuda()
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  # keep on GPU for fast swaps

dt.apply_delta_(model, "math.wdelta")
...
dt.swap_delta_(model, "code.wdelta", base_state)
```

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
    "version": 2,                # .wdelta format version
    "parent_hash": "e1810a...",  # SHA-256 of the base model
    "strategy": "int4",
    "n_tensors": 290,
    "tensors": {
        "model.embed_tokens.weight": {"shape": [151936, 896], "dtype": "bfloat16"},
        ...
    }
}
```

`dtype` is the base tensor's dtype for v2 files and `float32` (the stored delta dtype) for v1 files, which don't record it.

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
    verify="cached",
    num_workers=None,
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
| `verify` | `str \| bool` | [Verify mode](#verify-modes) for the base folder; later links are always hashed in memory unless `"none"`/`False` |
| `num_workers` | `int \| None` | Threads for the streaming first step |

**Returns:** Reconstructed state dict at the end of the chain, as float32 (chain hashes are defined over float32 models).

Applying deltas in the wrong order raises `ValueError: hash mismatch` immediately. Passing a directory for `base` uses the streaming loader for the first step.

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
| `state_dict` | `Dict[str, np.ndarray \| Tensor]` | State dict to hash. torch bfloat16 tensors are hashed by their raw 16-bit patterns, as stored in safetensors |

**Returns:** 64-character SHA-256 hex string.

Useful for verifying a chain link manually:

```python
v1_sd = dt.load_delta_from_paths("v1.wdelta", "base/", dtype="float32")
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
