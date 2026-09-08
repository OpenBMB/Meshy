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

    def build_sample(self, messages: Iterable[dict[str, str]], logprob: float = 0.0) -> Sample:
        sample = Sample(
            messages=[],
            tokens=[],
            logprobs=[],
            masks=[],
            ground_truth=None,
            reward=None,
            advantage=None
        )
        for msg in messages:
            sample = self.append_text(sample, msg["role"], msg["content"], logprob)
        return sample

    def append_text(self, sample: Sample, role: str, content: str, logprob: float = 0.0) -> Sample:
        sample.messages.append({"role": role, "content": content})
        if len(sample.messages) == 1:
            ids = self.tokenizer.apply_chat_template(sample.messages, tokenize=True, add_generation_prompt=True)["input_ids"]
        else:
            old_ids = self.tokenizer.apply_chat_template(sample.messages[:-1], tokenize=True, add_generation_prompt=True)["input_ids"]
            new_ids = self.tokenizer.apply_chat_template(sample.messages, tokenize=True, add_generation_prompt=True)["input_ids"]
            ids = new_ids[len(old_ids):]
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
