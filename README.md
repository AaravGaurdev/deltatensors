# deltatensors

**Lossless delta-compressed weight format for fine-tuned neural network models.**

Instead of storing fifty 15 GB fine-tunes of the same base model, store one base and fifty small `.wdelta` delta files. `deltatensors` handles the compression, and nearly exact reconstruction.


Example case would be something like:

```
base_model.safetensors   15 GB
checkpoint_01.wdelta        120 MB  
checkpoint_02.wdelta        118 MB
...
checkpoint_50.wdelta        115 MB
─────────────────────────────────
Total                    21 GB    vs  750 GB naive
```

## Install

```bash
pip install deltatensors
```

## Quick start

- TODO

## License

MIT
