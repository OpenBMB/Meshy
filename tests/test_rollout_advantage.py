from types import SimpleNamespace

import pytest

from meshy.advantage import _2_6_math_reshaped_advantage, _first_response_length
from meshy.worker.rollout import grpo_advantage, is_zero_variance_group


def _samples(*rewards):
    return [SimpleNamespace(reward=reward, advantage=None) for reward in rewards]


def test_default_advantage_is_grpo():
    samples = _samples(1.0, 2.0, 3.0)

    grpo_advantage(samples)

    assert [sample.advantage for sample in samples] == pytest.approx([-1.0, 0.0, 1.0])


def test_advantage_plugin_mutates_samples():
    samples = _samples(1.0, 2.0)

    def mutate(group):
        for sample in group:
            sample.advantage = -sample.reward

    mutate(samples)

    assert [sample.advantage for sample in samples] == [-1.0, -2.0]


def _masked_samples(rewards, lengths, *, later_span=0):
    return [
        SimpleNamespace(
            reward=reward,
            advantage=None,
            masks=[0, 0] + [1] * length + [0] + [1] * later_span,
        )
        for reward, length in zip(rewards, lengths)
    ]


def test_response_length_uses_first_contiguous_one_span():
    assert _first_response_length([0, 0, 1, 1, 1, 0, 1, 1]) == 3


def test_2_6_math_soft_overlong_penalty_and_group_center():
    samples = _masked_samples([1.0, 0.0], [80, 100], later_span=50)

    _2_6_math_reshaped_advantage(
        samples,
        rollout_max_response_len=100,
        overlong_buffer_len=20,
        overlong_penalty_factor=1.0,
        length_reward_weight=0.0,
    )

    assert [sample.advantage for sample in samples] == pytest.approx([1.0, -1.0])


def test_2_6_math_centers_without_std_normalization():
    samples = _masked_samples([1.0, 1.0, 0.0, 0.0], [10, 10, 10, 10])

    _2_6_math_reshaped_advantage(
        samples,
        overlong_buffer_len=0,
        length_reward_weight=0.0,
    )

    assert [sample.advantage for sample in samples] == [0.5, 0.5, -0.5, -0.5]


def test_2_6_math_optional_group_length_reward():
    samples = _masked_samples([1.0, 1.0, 0.0], [10, 30, 20])

    _2_6_math_reshaped_advantage(
        samples,
        overlong_buffer_len=0,
        length_reward_weight=0.2,
        length_reward_min_spread=1,
        length_reward_budget_floor=0,
    )

    assert [sample.advantage for sample in samples] == pytest.approx(
        [1.1 - 2 / 3, 0.9 - 2 / 3, -2 / 3]
    )


def test_zero_variance_group_filter():
    assert is_zero_variance_group(_samples(0.0, 0.0, 0.0))
    assert is_zero_variance_group(_samples(1.0, 1.0))
    assert not is_zero_variance_group(_samples(0.0, 1.0))
