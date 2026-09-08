"""Numerical parity between HF ``LlamaForCausalLM`` and the TorchTitan MiniCPM5.

Loads the real ``openbmb/MiniCPM5-1B`` checkpoint through
``MiniCPM5StateDictAdapter`` and compares fp32 logits against transformers.
Skipped unless a CUDA GPU is present and the checkpoint is already in the
local HF cache (no download is attempted).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

MODEL_ID = "openbmb/MiniCPM5-1B"


def _cached_snapshot() -> str | None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(MODEL_ID, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def test_minicpm5_1b_logits_match_hf():
    snapshot = _cached_snapshot()
    if snapshot is None:
        pytest.skip(f"{MODEL_ID} is not in the local HF cache")

    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from meshy.backend.titan.models.minicpm5 import (
        MiniCPM5StateDictAdapter,
        model_registry,
    )

    device = torch.device("cuda")
    spec = model_registry("1B")
    with device:
        model = spec.model.build()
    model.init_states(buffer_device=device)

    hf_state_dict = load_file(f"{snapshot}/model-00000-of-00001.safetensors")
    adapter = MiniCPM5StateDictAdapter(spec.model, snapshot)
    model.load_state_dict(
        {k: v.to(device) for k, v in adapter.from_hf(dict(hf_state_dict)).items()},
        strict=True,
    )
    model = model.float().eval()

    hf_model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32)
    hf_model = hf_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    ids = tokenizer(
        ["The capital of France is Paris, and the capital of Germany is"],
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        ref = hf_model(input_ids=ids).logits
        out = model(ids)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)
    assert torch.equal(out.argmax(-1), ref.argmax(-1))

    # Exporting back to HF layout must reproduce the original checkpoint.
    exported = adapter.to_hf(model.state_dict())
    for key, value in hf_state_dict.items():
        torch.testing.assert_close(exported[key].to(value.dtype).cpu(), value)
