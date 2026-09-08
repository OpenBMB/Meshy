"""Static checks for the MiniCPM5 TorchTitan model registry."""

import torch

from meshy.config import TrainerConfig
from meshy.backend.titan.config import build_forge_config
from meshy.backend.titan.models.minicpm5 import MiniCPM5StateDictAdapter, model_registry


def test_minicpm5_2b_matches_hf_architecture():
    spec = model_registry("2B")
    config = spec.model
    attention = config.layers[0].attention
    feed_forward = config.layers[0].feed_forward

    assert spec.name == "minicpm5"
    assert spec.state_dict_adapter is MiniCPM5StateDictAdapter
    assert config.dim == 2048
    assert config.vocab_size == 130560
    assert config.enable_weight_tying is False
    assert len(config.layers) == 42
    assert attention.n_heads == 16
    assert attention.n_kv_heads == 2
    assert attention.head_dim == 128
    assert attention.qkv_linear.wq.bias is False
    assert attention.qkv_linear.wkv.bias is False
    assert feed_forward.w1.out_features == 6144
    assert config.norm.eps == 1e-6
    assert config.rope.max_seq_len == 131072
    assert config.rope.theta == 5_000_000
    assert config.rope.scaling == "none"


def test_minicpm5_1b_matches_hf_architecture():
    spec = model_registry("1B")
    config = spec.model
    attention = config.layers[0].attention
    feed_forward = config.layers[0].feed_forward

    assert spec.name == "minicpm5"
    assert spec.flavor == "1B"
    assert config.dim == 1536
    assert config.vocab_size == 130560
    assert config.enable_weight_tying is False
    assert len(config.layers) == 24
    assert attention.n_heads == 16
    assert attention.n_kv_heads == 2
    assert attention.head_dim == 128
    assert attention.qkv_linear.wq.out_features == 2048
    assert attention.qkv_linear.wkv.out_features == 256
    assert attention.qkv_linear.wq.bias is False
    assert attention.qkv_linear.wkv.bias is False
    assert feed_forward.w1.out_features == 4608
    assert feed_forward.w2.in_features == 4608
    assert config.norm.eps == 1e-6
    assert config.rope.max_seq_len == 131072
    assert config.rope.theta == 5_000_000
    assert config.rope.scaling == "none"


def test_build_forge_config_resolves_minicpm5_and_hf_load(monkeypatch, tmp_path):
    # The runtime HF export directory must never override TorchTitan's native
    # DCP folder. This regression reproduces c059097's broken wiring.
    monkeypatch.setenv("XRL_CHECKPOINT_DIR", str(tmp_path / "hf-weights"))
    config = build_forge_config(
        TrainerConfig(
            model_name="minicpm5",
            model_flavor="2B",
            seq_len=16,
        ),
        hf_model_path="/tmp/minicpm5",
    )

    assert config.model_spec.name == "minicpm5"
    assert config.model_spec.flavor == "2B"
    assert config.hf_assets_path == "/tmp/minicpm5"
    assert config.checkpoint.enable is True
    assert config.checkpoint.folder == "checkpoint"
    assert config.checkpoint.initial_load_path == "/tmp/minicpm5"
    assert config.checkpoint.initial_load_in_hf is True


def test_rope_permute_roundtrip_on_gqa_kv_shape():
    """The failing load path views wk as [n_kv_heads=2, 64, 2, 2048]."""
    adapter = MiniCPM5StateDictAdapter(model_registry("2B").model, None)
    n_kv_heads = 2
    dim1, dim2 = 256, 2048
    weight = torch.randn(dim1, dim2)
    restored = adapter._reverse_permute(
        adapter._permute(weight, n_kv_heads, dim1, dim2),
        n_kv_heads,
        dim1,
        dim2,
    )
    torch.testing.assert_close(restored, weight)


def test_1b_hf_roundtrip_uses_actual_kv_width():
    """MiniCPM5-1B has head_dim=128 although dim // n_heads is 96.

    Upstream Llama3 sizes the K permute as ``n_kv_heads * (dim // n_heads)``
    (192 rows), which cannot view the real 256-row ``k_proj``.
    """
    adapter = MiniCPM5StateDictAdapter(model_registry("1B").model, None)
    hf_state_dict = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(2048, 1536),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(256, 1536),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(256, 1536),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(1536, 2048),
    }

    state_dict = adapter.from_hf(dict(hf_state_dict))

    assert state_dict["layers.0.attention.qkv_linear.wq.weight"].shape == (2048, 1536)
    assert state_dict["layers.0.attention.qkv_linear.wk.weight"].shape == (256, 1536)
    for key, value in adapter.to_hf(state_dict).items():
        torch.testing.assert_close(value, hf_state_dict[key])
