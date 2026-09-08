import re
from typing import Any

from meshy.dataset.base import Dataset
from meshy.utils.sample import Sample, SampleBuilder


def _extract_gsm8k_answer(response: str) -> float | None:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", response)
    if match is None:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


class GSM8K(Dataset):

    def __init__(self, batch_size: int, split: str = "train", hf_kwargs: dict = {}, seed: int | None = None):
        super().__init__(
            hf_kwargs={
                "path": "openai/gsm8k",
                "name": "main",
                "split": split,
                **hf_kwargs,
            },
            batch_size=batch_size,
            seed=seed
        )

    def apply_chat_template(self, data: dict, builder: SampleBuilder) -> Sample:
        sample = builder.build_sample([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data["question"] + " Let's think step by step and output the final answer after \"####\"."},
        ])
        sample.ground_truth = _extract_gsm8k_answer(data["answer"])
        return sample

    @staticmethod
    def reward(sample: Sample):
        predicted = _extract_gsm8k_answer(sample.messages[-1]["content"])
        if predicted is None:
            return 0.0
        else:
            return 1.0 if predicted == sample.ground_truth else 0.0
