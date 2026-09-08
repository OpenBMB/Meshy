"""``SGLangEngine.generate`` surfaces finish_reason and abort/resume cycles."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


def _engine_with_scripted_responses(responses):
    from meshy.engine.sglang import SGLangEngine

    engine = SGLangEngine("http://fake:1", {"max_new_tokens": 8})
    queue = list(responses)
    seen: list[dict] = []

    async def fake_post(url, payload, attempts):
        seen.append(payload)
        rows, reason = queue.pop(0)
        return SimpleNamespace(json=lambda: {"meta_info": {
            "output_token_logprobs": rows,
            "finish_reason": {"type": reason} if reason else None,
        }})

    engine._post = fake_post  # type: ignore[method-assign]
    return engine, seen


def test_generate_reports_truncation_and_continuations():
    engine, seen = _engine_with_scripted_responses([
        ([(-0.1, 11, ""), (-0.2, 12, "")], "abort"),   # colocate pause
        ([(-0.3, 13, "")], "length"),                   # resumed on new weights
    ])
    gen = asyncio.run(engine.generate([1, 2, 3]))
    tokens, logprobs = gen  # legacy unpacking still works
    assert tokens == [11, 12, 13]
    assert logprobs == pytest.approx([-0.1, -0.2, -0.3])
    assert gen.finish_reason == "length" and gen.truncated
    assert gen.continuations == 1
    # the continuation resubmits prompt + partial output with the remaining budget
    assert seen[1]["input_ids"] == [1, 2, 3, 11, 12]
    assert seen[1]["sampling_params"]["max_new_tokens"] == 6


def test_generate_stop_is_not_truncated():
    engine, _ = _engine_with_scripted_responses([([(-0.1, 5, "")], "stop")])
    gen = asyncio.run(engine.generate([1]))
    assert gen.finish_reason == "stop" and not gen.truncated and gen.continuations == 0


def test_generate_rejects_non_finite_logprobs():
    engine, _ = _engine_with_scripted_responses([([(float("nan"), 5, "")], "stop")])
    with pytest.raises(RuntimeError, match="non-finite"):
        asyncio.run(engine.generate([1]))
