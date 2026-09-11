from dataclasses import dataclass
from typing import Any, Iterable

from transformers import AutoTokenizer


@dataclass
class Sample:
    messages: Iterable[dict[str, str]]
    tokens: Iterable[int]
    logprobs: Iterable[float]
    masks: Iterable[int]
    ground_truth: Any
    reward: float
    advantage: float
    # Rollout-side quality stamps, filled by the rollout worker after the
    # response is generated (see :mod:`meshy.worker.rollout`).
    finish_reason: str | None = None
    #: the response hit ``max_new_tokens`` (SGLang ``finish_reason == "length"``)
    truncated: bool = False
    #: the response tail is degenerate repetition (:func:`meshy.utils.metric.has_repetition`)
    repetition: bool = False
    #: generation was aborted and resumed after a weight update, so the
    #: response mixes tokens from more than one policy version
    mixed_version: bool = False


class SampleBuilder:

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _render(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
        if not messages:
            return []
        out = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        # Some transformers versions return a BatchEncoding (a UserDict), others
        # a plain list of ids.
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def _spans(self, messages: list[dict[str, str]]) -> list[tuple[str | None, list[int]]]:
        """Split the rendered prompt into per-message ``(role, ids)`` spans.

        Spans are diffed on renders *without* the generation prompt: successive
        renders are then true prefixes of each other, so slicing by length is
        exact. (Diffing renders that carry the generation prompt is wrong: the
        prompt sits at the end of each render, so they are not prefix-related.)
        The final span is the generation prompt itself, with role ``None``.
        """
        spans: list[tuple[str | None, list[int]]] = []
        prev_ids: list[int] = []
        for i, msg in enumerate(messages):
            full_ids = self._render(messages[: i + 1], add_generation_prompt=False)
            spans.append((msg["role"], full_ids[len(prev_ids):]))
            prev_ids = full_ids
        tail = self._render(messages, add_generation_prompt=True)[len(prev_ids):]
        spans.append((None, tail))
        return spans

    def build_sample(self, messages: Iterable[dict[str, str]], logprob: float = 0.0) -> Sample:
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        sample = Sample(
            messages=messages,
            tokens=[],
            logprobs=[],
            masks=[],
            ground_truth=None,
            reward=None,
            advantage=None
        )
        for role, ids in self._spans(messages):
            sample.tokens.extend(ids)
            sample.logprobs.extend([logprob] * len(ids))
            sample.masks.extend([1 if role == "assistant" else 0] * len(ids))
        return sample

    def append_tokens(
        self,
        sample: Sample,
        role: str,
        tokens: Iterable[int],
        logprobs: Iterable[float],
    ) -> Sample:
        tokens = list(tokens)
        content = self.tokenizer.decode(tokens, skip_special_tokens=True)
        sample.messages.append({"role": role, "content": content})
        sample.tokens.extend(tokens)
        sample.logprobs.extend(logprobs)
        sample.masks.extend([1 if role == "assistant" else 0] * len(tokens))
        return sample
