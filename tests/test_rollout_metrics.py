"""Rollout metrics derived from the per-sample stamps (fixes after run 870776).

``repetition_frac`` / ``truncated_ratio`` / ``mixed_version_ratio`` come from
the stamps the rollout worker ships, not from token-id uniqueness or
``seq_len``; ``reward`` reaches the trainer so ``rollout/raw_reward_mean``
exists; staleness is measured against the trainer's current version.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from meshy.backend.titan.metrics import TitanTrainer


def _sample(
    length: int = 40,
    prompt: int = 10,
    *,
    reward: float = 1.0,
    advantage: float = 0.5,
    version: int = 3,
    truncated: bool = False,
    repetition: bool = False,
    mixed: bool = False,
    seed: int = 0,
) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(length)
    mask[:prompt] = 0
    logprobs = -torch.rand(length, generator=g)
    logprobs[:prompt] = 0.0  # prompt placeholders, like SampleBuilder.build_sample
    return TensorDict(
        {
            "tokens": torch.randint(1, 1000, (length,), generator=g),
            "logprobs": logprobs,
            "mask_assistant": mask,
            "advantage": torch.tensor(advantage),
            "weight_version": torch.tensor(version, dtype=torch.int64),
            "reward": torch.tensor(reward),
            "truncated": torch.tensor(int(truncated), dtype=torch.int64),
            "repetition": torch.tensor(int(repetition), dtype=torch.int64),
            "mixed_version": torch.tensor(int(mixed), dtype=torch.int64),
        },
        batch_size=[],
    )


def test_long_natural_text_is_not_flagged_by_token_uniqueness_anymore():
    """The old metric was ``1 - unique_tokens / tokens`` of the first sample."""
    tokens = torch.randint(1, 500, (30_000,)).tolist()  # 500-id "vocab", 30k tokens
    old_metric = 1.0 - len(set(tokens)) / len(tokens)
    assert old_metric > 0.9  # this is what run 870776 reported as "repetition"
    m, _ = TitanTrainer._rollout_metrics(
        [_sample(length=len(tokens), prompt=5, repetition=False)], seq_len=131072
    )
    assert m["rollout/repetition_frac"] == 0.0


def test_rollout_metrics_use_stamps_and_rewards():
    samples = [
        _sample(reward=1.0, version=3, truncated=True, repetition=False, mixed=True, seed=1),
        _sample(reward=0.0, version=3, truncated=False, repetition=True, mixed=False, seed=2),
        _sample(reward=1.0, version=2, truncated=False, repetition=False, mixed=False, seed=3),
        _sample(reward=0.0, version=1, truncated=True, repetition=False, mixed=True, seed=4),
    ]
    m, hist = TitanTrainer._rollout_metrics(samples, seq_len=131072, current_version=4)
    assert m["rollout/truncated_ratio"] == pytest.approx(0.5)
    assert m["rollout/repetition_frac"] == pytest.approx(0.25)
    assert m["rollout/weight_version/mixed_version_ratio"] == pytest.approx(0.5)
    # a per-sample fraction, not the old batch-level 0/1 indicator
    assert m["rollout/weight_version/stale_ratio"] == pytest.approx(0.5)
    assert m["rollout/weight_version/staleness_mean"] == pytest.approx((1 + 1 + 2 + 3) / 4)
    assert m["rollout/weight_version/staleness_max"] == 3
    assert m["rollout/raw_reward_mean"] == pytest.approx(0.5)
    assert m["rollout/rewards"] == pytest.approx(0.5)
    assert m["rollout/pass_rate"] == pytest.approx(0.5)
    # response length counts assistant tokens only
    assert m["rollout/response_len/mean"] == 30
    # a 40-token sample never reaches seq_len
    assert m["rollout/seq_len_truncated_ratio"] == 0.0
    # rollout log-probs average over sampled tokens only (prompt zeros excluded)
    assert m["rollout/log_probs"] < 0.0
    assert len(hist["rollout/log_probs"]) == 4 * 30


def test_rollout_metrics_without_stamps_omit_the_tags():
    plain = [
        TensorDict(
            {
                "tokens": torch.arange(20),
                "logprobs": -torch.ones(20),
                "mask_assistant": torch.ones(20),
                "advantage": torch.tensor(0.0),
                "weight_version": torch.tensor(0),
            },
            batch_size=[],
        )
    ]
    m, _ = TitanTrainer._rollout_metrics(plain, seq_len=16)
    for tag in ("rollout/truncated_ratio", "rollout/repetition_frac",
                "rollout/weight_version/mixed_version_ratio", "rollout/raw_reward_mean"):
        assert tag not in m
    assert m["rollout/seq_len_truncated_ratio"] == 1.0
