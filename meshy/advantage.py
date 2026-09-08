"""Pluggable group-level advantage functions."""

from __future__ import annotations

import os
from typing import Iterable


def _first_response_length(mask: Iterable[int]) -> int:
    """Return the length of the first contiguous run of ones in ``mask``."""
    length = 0
    in_response = False
    for value in mask:
        if value:
            in_response = True
            length += 1
        elif in_response:
            break
    return length


def _2_6_math_reshaped_advantage(
    samples: list,
    *,
    rollout_max_response_len: int = 126976,
    overlong_buffer_len: int | None = None,
    overlong_penalty_factor: float | None = None,
    length_reward_weight: float | None = None,
    length_reward_min_spread: int | None = None,
    length_reward_budget_floor: int | None = None,
) -> None:
    """Apply the DAPO soft-overlong pipeline and write advantages in place.

    Optional values default to the environment variables documented by the
    source pipeline. The group is centered without GRPO std normalization.
    """
    buffer_len = (
        int(os.environ.get("OVERLONG_BUFFER_LEN", "25395"))
        if overlong_buffer_len is None
        else int(overlong_buffer_len)
    )
    penalty_factor = (
        float(os.environ.get("OVERLONG_PENALTY_FACTOR", "1.0"))
        if overlong_penalty_factor is None
        else float(overlong_penalty_factor)
    )
    length_weight = (
        float(os.environ.get("LENGTH_REWARD_WEIGHT", "0"))
        if length_reward_weight is None
        else float(length_reward_weight)
    )
    min_spread = (
        int(os.environ.get("LENGTH_REWARD_MIN_SPREAD", "2000"))
        if length_reward_min_spread is None
        else int(length_reward_min_spread)
    )
    budget_floor = (
        int(os.environ.get("LENGTH_REWARD_BUDGET_FLOOR", "0"))
        if length_reward_budget_floor is None
        else int(length_reward_budget_floor)
    )

    raw_rewards = [float(sample.reward) for sample in samples]
    response_lengths = [_first_response_length(sample.masks) for sample in samples]
    shaped_rewards = list(raw_rewards)

    if buffer_len > 0:
        expected_len = rollout_max_response_len - buffer_len
        for i, response_length in enumerate(response_lengths):
            if response_length > expected_len:
                penalty = (
                    (expected_len - response_length)
                    / buffer_len
                    * penalty_factor
                )
                shaped_rewards[i] += max(penalty, -penalty_factor)

    if length_weight:
        correct = [i for i, reward in enumerate(raw_rewards) if reward > 0.5]
        if len(correct) >= 2:
            min_len = min(response_lengths[i] for i in correct)
            max_len = max(response_lengths[i] for i in correct)
            spread = max_len - min_len
            if spread >= min_spread and max_len >= budget_floor:
                for i in correct:
                    shaped_rewards[i] += length_weight * (
                        0.5 - (response_lengths[i] - min_len) / spread
                    )

    mean_reward = sum(shaped_rewards) / len(shaped_rewards)
    for sample, reward in zip(samples, shaped_rewards):
        sample.advantage = reward - mean_reward
