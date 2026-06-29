"""
HuggingFace Trainer integration for deltatensors.

Usage::

    from deltatensors.training import DeltaTensorsCallback

    callback = DeltaTensorsCallback(
        base_dir="qwen-base",
        strategy="int4",
        outlier_fraction=0.05,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[callback],
        ...
    )
    trainer.train()

After training, reconstruct any checkpoint::

    import deltatensors as dt

    state_dict = dt.load_delta_from_paths(
        "outputs/checkpoint-500/model.wdelta",
        "qwen-base",
    )
"""

from __future__ import annotations

import os
import glob as _glob
from pathlib import Path
from typing import Union

from .io import save_delta_from_paths

try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:
    # transformers is an optional dependency — define a no-op base so the class
    # can still be imported and used (HF Trainer uses duck typing anyway)
    class _TrainerCallback:  # type: ignore[no-redef]
        pass


class DeltaTensorsCallback(_TrainerCallback):
    """
    HuggingFace Trainer callback that saves each checkpoint as a ``.wdelta``
    delta file against a fixed base model.

    Called automatically by the Trainer after every checkpoint save.  The delta
    is written to ``{checkpoint_dir}/model.wdelta`` and is typically 3–5× smaller
    than the full model snapshot.

    Args:
        base_dir:               Path to the base model safetensors directory.
        strategy:               Compression strategy: ``"sparse"``, ``"quantized"``,
                                or ``"int4"`` (default, best quality).
        delete_full_checkpoint: If ``True``, remove ``.safetensors`` files from the
                                checkpoint directory after the delta is saved.
                                **Warning:** prevents resuming training from that
                                checkpoint and prevents HF ``load_best_model_at_end``.
                                Only set to ``True`` for checkpoints you will not
                                resume from.
        **strategy_kwargs:      Extra options forwarded to the compression strategy,
                                e.g. ``outlier_fraction=0.05`` for ``int4`` or
                                ``sparsity=0.9`` for ``sparse``.

    Note:
        ``use_gpu=False`` is forced during training because the GPU is already
        occupied by model weights, optimizer states, and activations.
        GPU-accelerated compression is available via ``save_delta_from_paths``
        when called standalone after training completes.

    Example::

        from deltatensors.training import DeltaTensorsCallback

        # checkpoints saved to outputs/checkpoint-N/model.wdelta
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
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        strategy: str = "int4",
        delete_full_checkpoint: bool = False,
        **strategy_kwargs,
    ):
        self.base_dir = str(base_dir)
        self.strategy = strategy
        self.delete_full_checkpoint = delete_full_checkpoint
        self.strategy_kwargs = strategy_kwargs

    def on_save(self, args, state, control, **kwargs):
        """Called by HuggingFace Trainer immediately after a checkpoint is saved."""
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        delta_path = os.path.join(checkpoint_dir, "model.wdelta")

        if not os.path.isdir(checkpoint_dir):
            print(f"[deltatensors] Warning: expected checkpoint dir not found: {checkpoint_dir}")
            return

        sf_files = _glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
        if not sf_files:
            print(
                f"[deltatensors] Warning: no .safetensors in {checkpoint_dir}; "
                "skipping delta save."
            )
            return

        try:
            save_delta_from_paths(
                delta_path,
                finetuned_dir=checkpoint_dir,
                base_dir=self.base_dir,
                strategy=self.strategy,
                use_gpu=False,  # GPU occupied by training
                **self.strategy_kwargs,
            )
        except Exception as exc:
            print(
                f"[deltatensors] Warning: delta save failed at step "
                f"{state.global_step}: {exc}"
            )
            return

        if self.delete_full_checkpoint:
            for sf in sf_files:
                os.remove(sf)
            print(
                f"[deltatensors] Removed full safetensors from {checkpoint_dir} "
                f"(delta: {os.path.basename(delta_path)})"
            )
