"""DAPO-Math-17k dataset for the worker-architecture Rollout.

Implements the worker ``Dataset`` API (``next_batch(builder)`` +
:class:`~meshy.utils.sample.Sample` + a static ``reward``); the prompt template
and boxed-answer scoring follow the DAPO recipe.
"""

from meshy.dataset.base import Dataset
from meshy.dataset.dapo_reward import compute_score
from meshy.utils.sample import Sample, SampleBuilder


class MATH(Dataset):

    def __init__(self, batch_size: int, hf_kwargs: dict = {}, seed: int | None = None):
        super().__init__(
            hf_kwargs={
                "path": "BytedTsinghua-SIA/DAPO-Math-17k",
                "split": "train",
                **hf_kwargs,
            },
            batch_size=batch_size,
            seed=seed,
        )

    def apply_chat_template(self, data: dict, builder: SampleBuilder) -> Sample:
        sample = builder.build_sample([
            {"role": "user", "content": data["prompt"][0]["content"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."},
        ])
        sample.ground_truth = data["reward_model"]["ground_truth"]
        return sample

    @staticmethod
    def reward(sample: Sample) -> float:
        result = compute_score(
            solution_str=sample.messages[-1]["content"],
            ground_truth=sample.ground_truth,
            strict_box_verify=True,
        )
        return float(result["score"])
