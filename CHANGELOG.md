# Changelog

## 0.3.0

Fast, memory-bounded reconstruction. Reconstruction results are unchanged
beyond float rounding: v1 files decode bit-identically to 0.2.0.

### Added

- `reconstruct_to_safetensors(base_dir, wdelta_path, out_dir, dtype="bfloat16", shard_size="5GB", verify="cached", num_workers=None, device="cpu")`:
  streams one tensor at a time into a (sharded) safetensors folder with
  `model.safetensors.index.json`, copies config/tokenizer files, and updates
  `config.json`'s `torch_dtype`. Peak RAM is about `num_workers` × the largest
  tensor in float32, independent of model size.
- `apply_delta_(model, wdelta_path, base_state=None)` and
  `swap_delta_(model, new_wdelta_path, base_state)`: apply or swap fine-tunes
  in a loaded torch model in place. Swaps restore from `base_state` rather than
  subtracting the old delta, because bfloat16 addition is not reversible.
- `verify="full" | "cached" | "none"` on the reconstruction APIs. `"cached"`
  (the new default) remembers verified hashes in `~/.cache/deltatensors`, keyed
  on path, size and mtime of every base shard.
- `num_workers` (default `min(8, cpu_count)`): thread pool across tensors;
  output is byte-identical for any worker count.
- `device="cuda"`: dequantize, add and cast on the GPU via torch;
  bit-identical to the numpy path.
- `decompress_add_` / `decompress_int4_add_`: add a delta into a preallocated
  float32 array in place.
- `benchmarks/bench_reconstruct.py` and `benchmarks/results.md`.

### Changed

- `.wdelta` format v2 (written by default; v1 still reads):
  - int4 packs all n values (outliers clipped, then overwritten on decode),
    stores `outlier_idx` as int32, and quantizes against the stored float16
    scale/zero point.
  - `parent_hash` covers tensors in sorted-name order for every writer.
  - Each tensor records `base_dtype`.
- int4 decoding uses a 256-entry lookup table gathered with `np.take` in
  cache-sized chunks: 3.6x (v1) to 5.7x (v2) faster than 0.2.0 on one core
  in `benchmarks/results.md`.
- `load_delta_from_paths` builds its result tensor by tensor instead of
  loading the whole base as float32 first, and returns the base's dtype
  (bfloat16 becomes float32 in numpy; `dtype="bfloat16"` returns torch
  tensors). New arguments: `verify` modes, `dtype`, `num_workers`, `device`.
- The `.wdelta` reader indexes records and reads arrays on demand, and
  checksums are streamed in chunks; no whole-file reads anywhere.
- Reading safetensors no longer needs torch (an internal reader handles
  bfloat16 as raw bits). The key→shard map is built once and shard handles
  stay open.
- `hash_state_dict` hashes torch bfloat16 tensors by their raw bits, matching
  the paths API.
- "Lossless" is gone from the package description: the format is
  near-lossless (int4 and quantized are lossy).

### Fixed

- Deltas saved with `save_delta_from_paths` from multi-shard models whose
  tensor names interleave across shards failed hash verification against an
  in-memory base (and against a base resharded differently).
- `save_delta_chain_from_paths` crashed (or mixed up tensors) when the parent
  delta came from such a multi-shard model; it read parent records by
  position instead of by name.
- int4 compression crashed on 1-element tensors.
- `save_delta` crashed on torch bfloat16 state dicts.
