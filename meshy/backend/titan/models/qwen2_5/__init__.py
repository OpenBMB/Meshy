# Copyright (c) Meshy project. Adapted from
# torchtitan/models/qwen3/__init__.py (BSD-style license).
"""Qwen2.5 model registry for torchtitan / ForgeEngine.

Architecturally Qwen2.5 is a very close cousin of Qwen3-dense:

* Same GQA grouped-query attention, same SwiGLU MLP, same RMSNorm
  (``eps = 1e-6``), same ``rope_theta = 1e6``, same packed-causal mask.
* **Qwen2.5 keeps biases on the QKV projections** (``q_proj.bias`` /
  ``k_proj.bias`` / ``v_proj.bias``); Qwen3 drops them.
* **Qwen2.5 has no QK-Norm**; Qwen3 normalizes Q/K with per-head RMSNorm
  before RoPE.
* **Weight tying for Qwen2.5 is on for 0.5B/1.5B/3B and off for >=7B**;
  Qwen3 ties small dense models too but stops at 4B.

Both differences are already first-class switches in torchtitan's
common building blocks:

* ``Linear.Config(bias=True)`` on the ``QKVLinear`` projections enables
  the QKV bias.
* Omitting ``qk_norm`` in ``GQAttention.Config`` leaves ``q_norm`` /
  ``k_norm`` ``None``, which the forward path interprets as "skip
  QK-Norm".

That means we do **not** need to fork ``Qwen3Model`` / ``GQAttention`` /
``parallelize_qwen3`` — we just feed them slightly different ``Config``
instances here, and pair the result with a state-dict adapter that
knows about the extra QKV bias keys (see ``state_dict_adapter.py``).

Config style follows torchtitan's current model registries: every
dimensional field is fully specified at config-construction time and the
per-layer configs are built eagerly into ``Qwen3Model.Config.layers``
(there is no layer template / deferred-init resolution any more).
"""

from dataclasses import replace
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RoPE
from torchtitan.models.common.attention import (
    FlexAttention,
    GQAttention,
    QKVLinear,
    ScaledDotProductAttention,
    VarlenAttention,
)
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.qwen3 import Qwen3Model, Qwen3TransformerBlock
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.protocols.model_spec import ModelSpec

from .state_dict_adapter import Qwen2D5StateDictAdapter

__all__ = [
    "qwen2_5_configs",
    "model_registry",
    "Qwen2D5StateDictAdapter",
]

_EPS = 1e-6


# ----------------------------------------------------------------------
# Param init recipes (mirror Qwen3, with one tweak)
# ----------------------------------------------------------------------
#
# Qwen2.5's QKV linears carry a bias, so the init dicts must contain a
# ``"bias"`` entry. ``_init_param`` iterates over the module's
# ``named_parameters(recurse=False)``, which for ``bias=False`` linears
# yields only ``weight`` and silently ignores extra dict keys — so the
# same dict is safe for the bias-free output / MLP linears too.
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


def _depth_init(layer_id: int):
    """Depth-scaled init for the residual-output projections (wo / w2 / w3)."""
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _output_linear_init(dim: int):
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _qwen2_5_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


# ----------------------------------------------------------------------
# Inner-attention backend selection
# ----------------------------------------------------------------------
def _attention_config(backend: str):
    """Resolve a backend name into ``(inner_attention config, mask_type)``.

    Same set of names as torchtitan's ``get_attention_config``; kept local
    because the ``flex_flash`` block size is picked per compute capability
    (Blackwell tolerates the wider 256x128 tile, Hopper does not).
    """
    match backend:
        case "sdpa":
            return ScaledDotProductAttention.Config(), "causal"
        case "flex":
            return FlexAttention.Config(), "block_causal"
        case "flex_flash":
            from torchtitan.tools.utils import has_cuda_capability

            if has_cuda_capability(10, 0):
                block_size = (256, 128)
            elif has_cuda_capability(9, 0):
                block_size = (128, 128)
            else:
                raise ValueError(
                    "Flash backend of FlexAttention is only supported on "
                    "Hopper or Blackwell"
                )
            return (
                FlexAttention.Config(
                    block_size=block_size, kernel_options={"BACKEND": "FLASH"}
                ),
                "block_causal",
            )
        case "varlen":
            return VarlenAttention.Config(), "block_causal"
        case _:
            raise ValueError(f"Invalid attention backend: {backend}")


# ----------------------------------------------------------------------
# Per-layer config builder
# ----------------------------------------------------------------------
def _build_qwen2_5_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    attn_backend: str,
) -> list[Qwen3TransformerBlock.Config]:
    """Build the per-layer configs (dense, no QK-Norm, with QKV bias)."""
    inner_attention, mask_type = _attention_config(attn_backend)

    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen2_5_norm(dim),
                ffn_norm=_qwen2_5_norm(dim),
                attention=GQAttention.Config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    qkv_linear=QKVLinear.Config(
                        head_dim=head_dim,
                        wq=Linear.Config(
                            in_features=dim,
                            out_features=n_heads * head_dim,
                            bias=True,
                            param_init=_LINEAR_INIT,
                        ),
                        # ``QKVLinear`` builds wk and wv from this one config.
                        wkv=Linear.Config(
                            in_features=dim,
                            out_features=n_kv_heads * head_dim,
                            bias=True,
                            param_init=_LINEAR_INIT,
                        ),
                    ),
                    wo=Linear.Config(
                        in_features=n_heads * head_dim,
                        out_features=dim,
                        param_init=_depth_init(layer_id),
                    ),
                    # Per-layer copy: ``update_from_config`` mutates the
                    # attention configs (sharding / CP checks), so layers must
                    # not alias one shared inner-attention config.
                    inner_attention=replace(inner_attention),
                    mask_type=mask_type,
                    rope_backend="cos_sin",
                    # qk_norm intentionally omitted: Qwen2.5 has no QK-Norm.
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=intermediate_size,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


# ----------------------------------------------------------------------
# Shared model-config builder
# ----------------------------------------------------------------------
def _qwen2_5_model(
    *,
    dim: int,
    n_layers: int,
    vocab_size: int,
    head_dim: int,
    n_heads: int,
    n_kv_heads: int,
    intermediate_size: int,
    enable_weight_tying: bool,
    max_seq_len: int = 32768,
    rope_theta: float = 1000000.0,
    attn_backend: str = "sdpa",
) -> Qwen3Model.Config:
    """Compose a full Qwen2.5 model config from per-flavor scalars.

    Small sizes (0.5B/1.5B/3B) use the SKIP embedding initializer and
    rely on weight tying to fill ``tok_embeddings.weight`` from
    ``lm_head.weight``. Larger sizes initialise the embedding directly
    (no tying).
    """
    tok_init = _EMBEDDING_SKIP_INIT if enable_weight_tying else _EMBEDDING_INIT
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen2_5_norm(dim),
        enable_weight_tying=enable_weight_tying,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=tok_init,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            backend="cos_sin",
        ),
        layers=_build_qwen2_5_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            attn_backend=attn_backend,
        ),
    )


# ----------------------------------------------------------------------
# Per-size factories
# ----------------------------------------------------------------------
#
# Architecture scalars below are taken from the HuggingFace config.json
# of the corresponding "Qwen/Qwen2.5-{flavor}-Instruct" checkpoint
# (rope_theta=1e6, rms_norm_eps=1e-6, hidden_act=silu, identical across
# the lineup). Cross-check before flipping a new flavor on for training.

def _debugmodel(attn_backend: str = "sdpa"):
    """Tiny model mainly for unit/smoke tests (~ same shape as Qwen3 debug)."""
    return _qwen2_5_model(
        dim=256,
        n_layers=8,
        vocab_size=2048,
        head_dim=128,
        n_heads=16,
        n_kv_heads=8,
        intermediate_size=3072,
        enable_weight_tying=True,
        max_seq_len=4096,
        attn_backend=attn_backend,
    )


def _0_5b(attn_backend: str = "sdpa"):
    return _qwen2_5_model(
        dim=896,
        n_layers=24,
        vocab_size=151936,
        head_dim=64,
        n_heads=14,
        n_kv_heads=2,
        intermediate_size=4864,
        enable_weight_tying=True,
        attn_backend=attn_backend,
    )


def _1_5b(attn_backend: str = "sdpa"):
    return _qwen2_5_model(
        dim=1536,
        n_layers=28,
        vocab_size=151936,
        head_dim=128,
        n_heads=12,
        n_kv_heads=2,
        intermediate_size=8960,
        enable_weight_tying=True,
        attn_backend=attn_backend,
    )


def _3b(attn_backend: str = "sdpa"):
    return _qwen2_5_model(
        dim=2048,
        n_layers=36,
        vocab_size=151936,
        head_dim=128,
        n_heads=16,
        n_kv_heads=2,
        intermediate_size=11008,
        enable_weight_tying=True,
        attn_backend=attn_backend,
    )


def _7b(attn_backend: str = "sdpa"):
    return _qwen2_5_model(
        dim=3584,
        n_layers=28,
        vocab_size=152064,
        head_dim=128,
        n_heads=28,
        n_kv_heads=4,
        intermediate_size=18944,
        enable_weight_tying=False,
        attn_backend=attn_backend,
    )


def _14b(attn_backend: str = "sdpa"):
    return _qwen2_5_model(
        dim=5120,
        n_layers=48,
        vocab_size=152064,
        head_dim=128,
        n_heads=40,
        n_kv_heads=8,
        intermediate_size=13824,
        enable_weight_tying=False,
        attn_backend=attn_backend,
    )


def _32b(attn_backend: str = "sdpa"):
    return _qwen2_5_model(
        dim=5120,
        n_layers=64,
        vocab_size=152064,
        head_dim=128,
        n_heads=40,
        n_kv_heads=8,
        intermediate_size=27648,
        enable_weight_tying=False,
        attn_backend=attn_backend,
    )


def _72b(attn_backend: str = "sdpa"):
    return _qwen2_5_model(
        dim=8192,
        n_layers=80,
        vocab_size=152064,
        head_dim=128,
        n_heads=64,
        n_kv_heads=8,
        intermediate_size=29568,
        enable_weight_tying=False,
        attn_backend=attn_backend,
    )


qwen2_5_configs = {
    "debugmodel": _debugmodel,
    "0.5B": _0_5b,
    "1.5B": _1_5b,
    "3B": _3b,
    "7B": _7b,
    "14B": _14b,
    "32B": _32b,
    "72B": _72b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
) -> ModelSpec:
    """Resolve a flavor string into a torchtitan ``ModelSpec`` for Qwen2.5.

    Mirrors the signature of ``torchtitan.models.qwen3.model_registry`` so
    that the dispatcher in ``meshy/backend/titan/config.py::_get_model_spec``
    can call both interchangeably.

    Args:
        flavor: Size key, one of :data:`qwen2_5_configs`.
        attn_backend: Inner-attention backend
            ("sdpa" | "flex" | "flex_flash" | "varlen").
    """
    if flavor not in qwen2_5_configs:
        raise ValueError(
            f"Unknown Qwen2.5 flavor '{flavor}'. "
            f"Available: {sorted(qwen2_5_configs)}"
        )

    config = qwen2_5_configs[flavor](attn_backend=attn_backend)

    return ModelSpec(
        name="qwen2_5",
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
