# deltatensors

**Near-lossless Post-Training delta compression for fine-tuned neural network models.**

Train however you want: full fine-tune, FSDP, whatever. Instead of storing 50 fine-tunes of the same base model, store one base and 50 small `.wdelta` delta files. `deltatensors` compresses the delta between a base and fine-tuned model, and reconstructs with sub-1% perplexity difference. 

**Tested on Qwen2.5-0.5B fine-tuned on WikiText-2:**
- Zero Noticeable Degradation: Perplexity: 19.11 (original) → 19.22 (reconstructed) — 0.58% perplexity difference
- Less degradation than standard int4 quantization of the full model
- 294 MB delta vs 953 MB fine-tuned model (3.2x)
- ~2.8x total storage reduction across 10 fine-tunes

```
base_model.safetensors   1.0 GB
checkpoint_01.wdelta     294 MB
checkpoint_02.wdelta     294 MB
...
checkpoint_10.wdelta     294 MB
─────────────────────────────────
Total                    3.9 GB    vs  11 GB naive
```

## Install

```bash
pip install deltatensors
pip install torch safetensors  # for loading from safetensors directories
```

Requires Python 3.9+.

## Quick start

```python
import deltatensors as dt

# save delta between a fine-tuned and base model (streaming, O(1) RAM)
dt.save_delta_from_paths("checkpoint.wdelta", "qwen-wiki/", "qwen-base/", strategy="int4")

# reconstruct without loading the full base into RAM
recon_sd = dt.load_delta_from_paths("checkpoint.wdelta", "qwen-base/")

# inspect a delta file without a base model
info = dt.inspect("checkpoint.wdelta")
# {'path': 'checkpoint.wdelta', 'size_mb': 294.2, 'strategy': 'int4', 'n_tensors': 290, ...}
```

## Compression strategies

| Strategy | Quality | Compression |
|---|---|---|
| `int4` | near-lossless (~0.5% PPL) | best |
| `sparse` | tunable via `sparsity=` | good |
| `quantized` | BitDelta-style 1-bit | aggressive |

`int4` uses outlier extraction (top k% weights stored as float16) + 4-bit quantization for the remainder — the strategy used in the benchmark above.

## HuggingFace Trainer integration

Drop `DeltaTensorsCallback` into any `Trainer` run to save each checkpoint as a `.wdelta` instead of (or alongside) the full safetensors snapshot:

```python
from deltatensors.training import DeltaTensorsCallback
from transformers import Trainer, TrainingArguments

callback = DeltaTensorsCallback(
    base_dir="path/to/base-model",   # the model you started from
    strategy="int4",
    outlier_fraction=0.05,
    delete_full_checkpoint=False,     # set True to save disk space (can't resume training)
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="outputs", save_steps=500, ...),
    callbacks=[callback],
    ...
)
trainer.train()
```

Each checkpoint gets a `model.wdelta` alongside the usual safetensors files. GPU compression is disabled during training (the GPU is occupied by optimizer states); it runs on CPU automatically.

Reconstruct any checkpoint afterwards:

```python
sd = dt.load_delta_from_paths("outputs/checkpoint-500/model.wdelta", "path/to/base-model")
```

## Lineage chains

Track and reconstruct a full fine-tuning history with chains of `.wdelta` files. Each delta in the chain stores its delta against the *prior* reconstructed model (not the original base), so incremental updates stay small.

```
base ──► v1.wdelta ──► v1_model ──► v2.wdelta ──► v2_model ──► ...
```

```python
# Save a chained delta (v2 vs reconstructed v1, not vs base)
dt.save_delta_chain_from_paths(
    "v2.wdelta",
    finetuned_dir="v2_checkpoint/",
    parent_delta_path="v1.wdelta",
    base_dir="base_model/",
    strategy="int4",
)

# Inspect the chain without loading any tensors
history = dt.inspect_chain(["v1.wdelta", "v2.wdelta", "v3.wdelta"])
for entry in history:
    print(entry["step"], entry["size_mb"], "MB", entry["parent_hash"][:8])

# Reconstruct the model at the end of the chain — verifies each hash link
sd = dt.load_delta_chain(["v1.wdelta", "v2.wdelta"], base="base_model/")
```

Each `.wdelta` records a `parent_hash` (SHA-256 of the model it was computed against). `load_delta_chain` verifies every link automatically — applying deltas in the wrong order raises `ValueError` immediately.

`save_delta_chain_from_paths` is fully streaming: peak RAM is O(one tensor pair), not O(two full models).

## Why not LoRA?

LoRA constrains the delta to be low-rank *during training*, which limits expressiveness. `deltatensors` compresses arbitrary full fine-tune deltas *after training* — no constraints on how you fine-tune.

## License

MIT

---
p.s. *If you find `deltatensors` useful, please consider leaving a ⭐ star on the repository to help others find it!*
