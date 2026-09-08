"""MiniCPM5 model registry for TorchTitan's ForgeEngine.

MiniCPM5-1B and MiniCPM5-2B are published as ``LlamaForCausalLM`` and use
the standard Llama parameter layout. Their dimensions do not match any
built-in TorchTitan Llama flavor, so this module supplies exact model configs
while reusing the upstream implementation and parallelization code. The
state-dict adapter is a DTensor-safe subclass of the Llama3 adapter (GQA
``n_kv_heads=2`` cannot be viewed in-place under a 4-way FSDP mesh).
"""

from functools import partial

import torch.nn as nn

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.llama3 import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.protocols.model_spec import ModelSpec

from .state_dict_adapter import MiniCPM5StateDictAdapter

__all__ = ["minicpm5_configs", "model_registry", "MiniCPM5StateDictAdapter"]

_EPS = 1e-6
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _depth_init(layer_id: int):
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _output_linear_init(dim: int):
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(
        normalized_shape=dim,
        eps=_EPS,
        param_init=_NORM_INIT,
    )


def _minicpm5_model(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    n_layers: int,
    head_dim: int,
    intermediate_size: int,
    vocab_size: int,
    max_seq_len: int = 131072,
    rope_theta: float = 5_000_000,
    attn_backend: str = "sdpa",
) -> Llama3Model.Config:
    """Build a MiniCPM5 dense config from the model's HF architecture values."""
    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = [
        Llama3TransformerBlock.Config(
            attention_norm=_norm(dim),
            ffn_norm=_norm(dim),
            attention=make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                mask_type=mask_type,
                rope_backend="complex",
            ),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=intermediate_size,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            ),
        )
        for layer_id in range(n_layers)
    ]

    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=_norm(dim),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            backend="complex",
            scaling="none",
        ),
        layers=layers,
    )


def _minicpm5_1b(attn_backend: str = "sdpa") -> Llama3Model.Config:
    """Build the exact architecture from ``openbmb/MiniCPM5-1B``."""
    return _minicpm5_model(
        dim=1536,
        n_heads=16,
        n_kv_heads=2,
        n_layers=24,
        head_dim=128,
        intermediate_size=4608,
        vocab_size=130560,
        attn_backend=attn_backend,
    )


def _minicpm5_2b(attn_backend: str = "sdpa") -> Llama3Model.Config:
    """Build the exact architecture from ``openbmb/MiniCPM5-2B``."""
    return _minicpm5_model(
        dim=2048,
        n_heads=16,
        n_kv_heads=2,
        n_layers=42,
        head_dim=128,
        intermediate_size=6144,
        vocab_size=130560,
        attn_backend=attn_backend,
    )


minicpm5_configs = {
    "1B": _minicpm5_1b,
    "2B": _minicpm5_2b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    quantization: list[QuantizationConverter.Config] | None = None,
) -> ModelSpec:
    """Resolve a MiniCPM5 flavor into a TorchTitan ``ModelSpec``."""
    if flavor not in minicpm5_configs:
        raise ValueError(
            f"Unknown MiniCPM5 flavor '{flavor}'. "
            f"Available: {sorted(minicpm5_configs)}"
        )

    model = minicpm5_configs[flavor](attn_backend=attn_backend)
    if quantization is not None:
        for converter in quantization:
            converter.build().convert(model)

    return ModelSpec(
        name="minicpm5",
        flavor=flavor,
        model=model,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=MiniCPM5StateDictAdapter,
    )
