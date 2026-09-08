# Copyright (c) Meshy project. Adapted from
# torchtitan/models/qwen3/state_dict_adapter.py (BSD-style license).
"""HF<->torchtitan state-dict adapter for Qwen2.5.

Re-uses the Qwen3 adapter as a base — both models share the dense Qwen
layout (``model.layers.{i}.self_attn.{q,k,v,o}_proj.weight``,
``mlp.{gate,up,down}_proj.weight``, ``input_layernorm`` / ``post_attention_layernorm``,
``model.embed_tokens.weight``, ``lm_head.weight``) — and only patches
two things:

1. **Adds** mappings for ``q_proj.bias`` / ``k_proj.bias`` / ``v_proj.bias``
   (Qwen2.5 has biases on QKV; Qwen3 does not).
2. **Removes** the ``q_norm.weight`` / ``k_norm.weight`` mappings
   (Qwen2.5 has no QK-Norm; Qwen3 does).

Weight tying (small flavors 0.5B/1.5B/3B) is already handled in the
parent class's ``from_hf`` path, which fills in a synthetic
``lm_head.weight`` from ``model.embed_tokens.weight`` when the HF
checkpoint omits the LM head (the standard tied-weight layout).
"""

from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter


class Qwen2D5StateDictAdapter(Qwen3StateDictAdapter):
    """Qwen2.5 variant of :class:`Qwen3StateDictAdapter`."""

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)

        self.from_hf_map.update(
            {
                "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.qkv_linear.wq.bias",
                "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.qkv_linear.wk.bias",
                "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.qkv_linear.wv.bias",
            }
        )

        for hf_key in (
            "model.layers.{}.self_attn.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight",
        ):
            self.from_hf_map.pop(hf_key, None)
