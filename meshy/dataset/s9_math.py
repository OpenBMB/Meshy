"""Local JSONL mathematics dataset used by the long-context recipe."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from meshy.dataset.math_reward import math_reward
from meshy.utils.sample import Sample, SampleBuilder


class S9Math:
    """Read S9 records with ``prompt`` messages and a scalar ``label``."""

    def __init__(
        self,
        batch_size: int,
        path: str | None = None,
        seed: int | None = None,
        prompt_max_tokens: int = 2048,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        path = path or os.environ.get("S9_DATASET_PATH")
        if not path:
            raise RuntimeError("S9_DATASET_PATH must point to the S9 JSONL dataset")
        self.batch_size = int(batch_size)
        self.prompt_max_tokens = int(prompt_max_tokens)
        if self.prompt_max_tokens <= 0:
            raise ValueError("prompt_max_tokens must be positive")
        self.records = self._read(Path(path))
        if seed is not None:
            random.Random(seed).shuffle(self.records)
        self.index = 0

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"S9 JSONL dataset does not exist: {path}")
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on S9 line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"S9 line {line_number} must be a JSON object")
                if "prompt" not in record or "label" not in record:
                    raise ValueError(f"S9 line {line_number} requires prompt and label")
                records.append(record)
        if not records:
            raise ValueError(f"S9 JSONL dataset is empty: {path}")
        return records

    def next_batch(self, builder: SampleBuilder) -> list[Sample]:
        if self.index >= len(self.records):
            return []
        records = self.records[self.index : self.index + self.batch_size]
        self.index += len(records)
        return [self.apply_chat_template(record, builder) for record in records]

    def apply_chat_template(self, data: dict[str, Any], builder: SampleBuilder) -> Sample:
        messages = data["prompt"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("S9 prompt must be a non-empty message list")
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("S9 prompt messages must be objects")
            role, content = message.get("role"), message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("S9 prompt messages require string role/content")
            normalized.append({"role": role, "content": content})
        sample = builder.build_sample(normalized)
        if len(list(sample.tokens)) > self.prompt_max_tokens:
            raise ValueError(
                f"S9 prompt exceeds {self.prompt_max_tokens} tokens "
                f"(got {len(list(sample.tokens))})"
            )
        sample.ground_truth = data["label"]
        return sample

    @staticmethod
    def reward(sample: Sample) -> float:
        return math_reward(sample.messages[-1]["content"], sample.ground_truth)


__all__ = ["S9Math"]
