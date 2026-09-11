"""SampleBuilder must produce exactly the tokens a one-shot chat-template
render with the generation prompt would produce.

The regression guarded here: building the prompt incrementally by diffing
renders that each carry the generation prompt corrupts multi-turn prompts
(the ``<|im_start|>user`` header of the second message is replaced by a
stray ``<|im_start|>assistant`` header), while keeping the token count
identical — a silent failure. Uses a deterministic ChatML-style fake
tokenizer, so no model download is needed.
"""

from __future__ import annotations

import types

import pytest


class _ChatMLTokenizer:
    """Character-level tokenizer with a ChatML-style chat template."""

    def _render_text(self, messages, add_generation_prompt):
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = self._render_text(messages, add_generation_prompt)
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, tokens, **kw):
        return "".join(chr(t) for t in tokens)


@pytest.fixture()
def builder(monkeypatch):
    import meshy.utils.sample as sample_mod

    monkeypatch.setattr(
        sample_mod,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: _ChatMLTokenizer()),
    )
    return sample_mod.SampleBuilder("stub-model")


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "What is 2+2?"}],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
    ],
    ids=["user", "system+user", "multi-turn"],
)
def test_build_sample_matches_full_render(builder, messages):
    sample = builder.build_sample(messages)
    expected = builder.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )["input_ids"]
    assert sample.tokens == expected
    assert len(sample.logprobs) == len(sample.tokens)
    assert len(sample.masks) == len(sample.tokens)


def test_masks_cover_assistant_spans_only(builder):
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "V"},
    ]
    sample = builder.build_sample(messages)
    tok = builder.tokenizer
    masked = tok.decode([t for t, m in zip(sample.tokens, sample.masks) if m == 1])
    assert masked == "<|im_start|>assistant\nA<|im_end|>\n"
    # The trailing generation prompt is part of the prompt, not the response.
    assert sample.masks[-1] == 0


def test_render_accepts_plain_list_return(builder):
    """apply_chat_template returning a bare list (older transformers) works too."""
    tok = builder.tokenizer
    orig = tok.apply_chat_template
    tok.apply_chat_template = lambda *a, **k: list(orig(*a, **k)["input_ids"])
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    sample = builder.build_sample(messages)
    expected = orig(messages, tokenize=True, add_generation_prompt=True)["input_ids"]
    assert sample.tokens == expected
