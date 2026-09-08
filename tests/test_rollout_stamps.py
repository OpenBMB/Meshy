"""Per-sample rollout quality stamps: ``reward`` / ``truncated`` / ``repetition`` /
``mixed_version`` are computed by the rollout worker and shipped as TQ columns."""

from __future__ import annotations

import torch

from meshy.utils.metric import compression_ratio, has_repetition


def test_has_repetition_flags_loops_not_prose():
    import random

    rng = random.Random(0)
    words = ["theorem", "let", "x", "be", "prime", "then", "sum", "over", "k", "modulo", "hence", "we"]
    prose = " ".join(rng.choice(words) + str(rng.randint(0, 999)) for _ in range(6000))
    assert len(prose) > 10_000
    assert not has_repetition(prose)
    loop = "Therefore the answer is 42. " * 2000
    assert has_repetition(loop)
    assert compression_ratio(loop) > 10
    # Short responses are never flagged, whatever their content.
    assert not has_repetition("abc " * 100)


def test_rollout_worker_stamps_and_tensordict_columns():
    from meshy.engine.sglang import Generation
    from meshy.utils.sample import Sample
    from meshy.worker.rollout import GRPO_FIELDS, RolloutWorker, sample_to_tensordict
    from meshy.config import GRPO_TRAINER_FIELDS
    from meshy.transferqueue import adapter

    assert tuple(GRPO_FIELDS) == tuple(GRPO_TRAINER_FIELDS)
    sample = Sample(
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "loop " * 5000}],
        tokens=[1, 2, 3], logprobs=[0.0, -0.5, -0.5], masks=[0, 1, 1],
        ground_truth=None, reward=1.0, advantage=0.25,
    )
    RolloutWorker._stamp_sample(sample, Generation([2, 3], [-0.5, -0.5], "length", continuations=2))
    assert sample.truncated and sample.mixed_version and sample.repetition
    td = sample_to_tensordict(sample, weight_version=7)
    assert set(GRPO_FIELDS) <= set(td.keys())
    assert td["reward"].item() == 1.0 and td["truncated"].item() == 1
    assert td["mixed_version"].item() == 1 and td["repetition"].item() == 1
    batched = adapter.samples_to_td([td, td], GRPO_FIELDS)
    back = adapter.td_to_samples(batched, GRPO_FIELDS)
    assert back[0]["truncated"].dtype == torch.int64 and back[1]["reward"].item() == 1.0

    # legacy (tokens, logprobs) tuples leave the flags at their defaults
    plain = Sample(messages=[{"role": "assistant", "content": "x"}], tokens=[1], logprobs=[0.0],
                   masks=[1], ground_truth=None, reward=0.0, advantage=0.0)
    RolloutWorker._stamp_sample(plain, ([1], [0.0]))
    assert not plain.truncated and not plain.mixed_version and not plain.repetition
