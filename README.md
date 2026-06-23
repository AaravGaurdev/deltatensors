# deltatensors

**Near-lossless delta compression for fine-tuned neural network models.**

Instead of storing 50 fine-tunes of the same base model, store one base and 50 small `.wdelta` delta files. `deltatensors` compresses the delta between a base and fine-tuned model, and reconstructs with sub-1% perplexity difference.

**Tested on Qwen2.5-0.5B fine-tuned on WikiText-2:**
- Perplexity: 19.11 (original) → 19.22 (reconstructed) — 0.58% perplexity difference
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

## Quick start

```python
import deltatensors as dt

# save delta between a fine-tuned and base model (streaming, O(1) RAM)
dt.save_delta_from_paths("checkpoint.wdelta", "qwen-wiki/", "qwen-base/", strategy="int4")

# reconstruct without loading the full base into RAM
recon_sd = dt.load_delta_from_paths("checkpoint.wdelta", "qwen-base/")

# inspect a delta file without a base model
info = dt.inspect("checkpoint.wdelta")
print(info)
# {'path': 'checkpoint.wdelta', 'size_mb': 294.2, 'strategy': 'int4', 'n_tensors': 290, ...}
```

## Compression strategies

| Strategy | Quality | Compression |
|---|---|---|
| `int4` | near-lossless (~0.5% PPL) | best |
| `sparse` | tunable via `sparsity=` | good |
| `quantized` | BitDelta-style 1-bit | aggressive |

*`int4` uses outlier extraction (top k% weights stored in float16) + 4-bit quantization for the remainder. This was the strategy used for the example at the start.*

## Why not LoRA?

LoRA constrains the delta to be low-rank *during training*, which limits expressiveness. `deltatensors` compresses arbitrary full fine-tune deltas *after training* - no constraints on how you fine-tune.

## Roadmap

- **Lineage** — chain multiple `.wdelta` files to track and reconstruct full fine-tuning histories

## License

MIT