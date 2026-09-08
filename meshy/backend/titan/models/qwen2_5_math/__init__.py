# Copyright (c) Meshy project. Adapted from the legacy Qwen2.5 registry.
"""Qwen2.5-Math + DeepSeek-R1-Distill-Qwen model registry.

Architecturally these checkpoints reuse the standard Qwen2.5 dense
layout (Qwen3 building blocks + QKV bias, no QK-Norm) — so we delegate
``_qwen2_5_model`` / ``Qwen2D5StateDictAdapter`` from the neighbouring
:mod:`qwen2_5` package and only override two hyperparameters that the
standard Qwen2.5 flavors get wrong for this family:

1. ``rope_theta = 10000.0`` (Qwen2.5-Math base; standard Qwen2.5 uses 1e6).
2. Per-flavor ``enable_weight_tying``:

   * ``Qwen2.5-Math-1.5B``                — tied.
   * ``Qwen2.5-Math-7B``                  — untied.
   * ``DeepSeek-R1-Distill-Qwen-1.5B``    — untied (independent ``lm_head``).
   * ``DeepSeek-R1-Distill-Qwen-7B``      — untied.

Getting either wrong produces gibberish: the wrong RoPE base scrambles
positional encoding by ~100×, and a stale ``enable_weight_tying=True``
collapses ``tok_embeddings.weight`` and ``output.weight`` into a single
``nn.Parameter`` so HF's two distinct tensors clobber each other at load
time.

Note: DeepSeek-R1-Distill-Qwen-{14B, 32B} inherit standard Qwen2.5
backbones (``rope_theta = 1e6``, untied) and should be loaded through
the plain :mod:`qwen2_5` registry, not this one.
"""

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.protocols.model_spec import ModelSpec

from ..qwen2_5 import _qwen2_5_model
from .state_dict_adapter import Qwen2D5StateDictAdapter

__all__ = [
    "qwen2_5_math_configs",
    "model_registry",
    "Qwen2D5StateDictAdapter",
]


# ----------------------------------------------------------------------
# Per-flavor factories
# ----------------------------------------------------------------------
#
# Scalars below are pulled verbatim from each model's HuggingFace
# ``config.json``. Cross-check before flipping a new flavor on.

def _math_1_5b(attn_backend: str = "sdpa"):
    """Qwen2.5-Math-1.5B (tied embeddings)."""
    return _qwen2_5_model(
        dim=1536,
        n_layers=28,
        vocab_size=151936,
        head_dim=128,
        n_heads=12,
        n_kv_heads=2,
        intermediate_size=8960,
        enable_weight_tying=True,
        rope_theta=10000.0,
        max_seq_len=4096,
        attn_backend=attn_backend,
    )


def _math_7b(attn_backend: str = "sdpa"):
    """Qwen2.5-Math-7B (untied embeddings)."""
    return _qwen2_5_model(
        dim=3584,
        n_layers=28,
        vocab_size=152064,
        head_dim=128,
        n_heads=28,
        n_kv_heads=4,
        intermediate_size=18944,
        enable_weight_tying=False,
        rope_theta=10000.0,
        max_seq_len=4096,
        attn_backend=attn_backend,
    )


def _r1_distill_1_5b(attn_backend: str = "sdpa"):
    """DeepSeek-R1-Distill-Qwen-1.5B (same shape as Math-1.5B but untied)."""
    return _qwen2_5_model(
        dim=1536,
        n_layers=28,
        vocab_size=151936,
        head_dim=128,
        n_heads=12,
        n_kv_heads=2,
        intermediate_size=8960,
        enable_weight_tying=False,
        rope_theta=10000.0,
        max_seq_len=131072,
        attn_backend=attn_backend,
    )


def _r1_distill_7b(attn_backend: str = "sdpa"):
    """DeepSeek-R1-Distill-Qwen-7B (same shape as Math-7B, longer context)."""
    return _qwen2_5_model(
        dim=3584,
        n_layers=28,
        vocab_size=152064,
        head_dim=128,
        n_heads=28,
        n_kv_heads=4,
        intermediate_size=18944,
        enable_weight_tying=False,
        rope_theta=10000.0,
        max_seq_len=131072,
        attn_backend=attn_backend,
    )


qwen2_5_math_configs = {
    "Math-1.5B": _math_1_5b,
    "Math-7B": _math_7b,
    "R1-Distill-1.5B": _r1_distill_1_5b,
    "R1-Distill-7B": _r1_distill_7b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
) -> ModelSpec:
    """Resolve a flavor string into a torchtitan ``ModelSpec``.

    Mirrors :func:`meshy.backend.titan.models.qwen2_5.model_registry` so that the
    dispatcher in ``meshy/backend/titan/config.py::_get_model_spec`` can call
    both interchangeably.

    Args:
        flavor: Size key, one of :data:`qwen2_5_math_configs`.
        attn_backend: Inner-attention backend
            ("sdpa" | "flex" | "flex_flash" | "varlen").
    """
    if flavor not in qwen2_5_math_configs:
        raise ValueError(
            f"Unknown Qwen2.5-Math flavor '{flavor}'. "
            f"Available: {sorted(qwen2_5_math_configs)}"
        )

    config = qwen2_5_math_configs[flavor](attn_backend=attn_backend)

    return ModelSpec(
        name="qwen2_5_math",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        # NOTE: no ``build_loss_fn`` — torchtitan moved the loss off
        # ``ModelSpec`` and onto the job config; see
        # :mod:`meshy.backend.titan.compat`.
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen2D5StateDictAdapter,
    )
