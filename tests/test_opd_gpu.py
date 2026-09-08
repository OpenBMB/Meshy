"""End-to-end Student Top-K OPD on one card with the Qwen3 debug model.

A frozen Teacher trainer scores the rows through the same planner / layout the
Student uses; the Student trainer then runs real ``train_step``s.  Skipped
without CUDA. Run on one card::

    CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_opd_gpu.py -q
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

SEQ_LEN = 512
LENGTHS = (300, 120, 50, 200, 33, 260)
TOP_K = 8


def _single_rank_env():
    for k, v in (("LOCAL_RANK", "0"), ("RANK", "0"), ("WORLD_SIZE", "1"),
                 ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29573")):
        os.environ.setdefault(k, v)


def _samples(seed: int = 0):
    from tensordict import TensorDict

    g = torch.Generator().manual_seed(seed)
    out = []
    for i, L in enumerate(LENGTHS):
        mask = torch.ones(L)
        mask[: L // 3] = 0
        out.append(TensorDict({
            "tokens": torch.randint(1, 2000, (L,), generator=g),
            "logprobs": -torch.rand(L, generator=g) * 2,
            "mask_assistant": mask,
            "advantage": torch.randn((), generator=g),
            "weight_version": torch.tensor(0, dtype=torch.int64),
            "reward": torch.tensor(float(i % 2)),
            "truncated": torch.tensor(0, dtype=torch.int64),
            "repetition": torch.tensor(0, dtype=torch.int64),
            "mixed_version": torch.tensor(0, dtype=torch.int64),
        }, batch_size=[]))
    return out


def _cfg(layout: str, lr: float, max_norm: float = 1e9):
    from meshy.config import TrainerConfig

    return TrainerConfig(
        model_name="qwen3", model_flavor="debugmodel", seq_len=SEQ_LEN,
        attn_backend="varlen" if layout == "packed" else "sdpa",
        dp_shard_degree=-1, enable_checkpoint=False, compile_model=False,
        dump_folder="/tmp/xrl_opd_test", lr=lr, max_norm=max_norm,
    )


def _teacher(seed: int):
    from meshy.backend.titan.config import build_forge_config
    from meshy.backend.titan.trainer import TitanTrainer

    torch.manual_seed(seed)
    return TitanTrainer(build_forge_config(_cfg("padded", 0.0)), batch_layout="padded",
                        micro_batch_size=3, mini_batch_size=6, seq_bucket=64, timer_enabled=False)


def _student(layout: str, seed: int, lr: float = 0.0, max_norm: float = 1e9, **params):
    from meshy.backend.titan.config import build_forge_config
    from meshy.backend.titan.opd import StudentTopKTrainer

    torch.manual_seed(seed)
    return StudentTopKTrainer(build_forge_config(_cfg(layout, lr, max_norm)), batch_layout=layout,
                              mini_batch_size=6, seq_bucket=64, timer_enabled=False,
                              tensorboard_enabled=False, **params)


def _copy_weights(src, dst):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, set_model_state_dict,
    )

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    set_model_state_dict(dst.model_parts[0], get_model_state_dict(src.model_parts[0], options=opts), options=opts)


def _scored(teacher, samples):
    from meshy.engine.opd import score_samples

    scored = score_samples(teacher, samples, TOP_K)
    assert scored.keys() == set(range(len(samples)))
    for i, td in enumerate(samples):
        ids, lps = scored[i]
        assert ids.shape == (LENGTHS[i], TOP_K) and lps.shape == (LENGTHS[i], TOP_K)
        td["teacher_topk_ids"] = ids
        td["teacher_topk_logprobs"] = lps
    return samples


@pytest.fixture(scope="module")
def models():
    _single_rank_env()
    return _teacher(seed=0), _student("padded", seed=1, micro_batch_size=3), _student("packed", seed=2, max_tokens_per_micro=640)


def test_teacher_topk_logprobs_are_a_distribution(models):
    teacher, _, _ = models
    samples = _scored(teacher, _samples())
    for td in samples:
        lps = td["teacher_topk_logprobs"]
        assert torch.all(lps <= 0) and torch.all(lps.exp().sum(-1) <= 1.0 + 1e-4)
        assert (lps[:, 0] >= lps[:, -1]).all()


def test_student_equal_to_teacher_has_zero_kl(models):
    teacher, padded, _ = models
    _copy_weights(teacher, padded)
    samples = _scored(teacher, _samples(seed=3))
    result = padded.train_step(samples)
    assert result["train/distill_loss"] < 5e-2, result["train/distill_loss"]
    assert abs(result["train/student_topk_mass"] - result["train/teacher_topk_mass"]) < 5e-2
    assert 0.0 < result["train/teacher_topk_mass"] <= 1.0
    assert "train/train_rollout_kl" in result and "train/grad_norm" in result


def test_padded_and_packed_students_agree(models):
    teacher, padded, packed = models
    _copy_weights(padded, packed)
    samples = _scored(teacher, _samples(seed=4))
    a = padded.train_step(samples)
    b = packed.train_step(samples)
    assert b["train/padding_ratio"] < a["train/padding_ratio"]
    assert abs(a["train/distill_loss"] - b["train/distill_loss"]) < 2e-2 * max(1.0, a["train/distill_loss"])
    assert abs(a["grad_norm"] - b["grad_norm"]) < 3e-2 * max(1.0, a["grad_norm"])


def test_student_learns_towards_a_different_teacher():
    _single_rank_env()
    teacher = _teacher(seed=10)
    student = _student("padded", seed=11, lr=3e-4, max_norm=1.0, micro_batch_size=3)
    samples = _scored(teacher, _samples(seed=5))
    losses = [student.train_step(samples)["train/distill_loss"] for _ in range(12)]
    assert losses[0] > 0.1 and min(losses[-3:]) < 0.7 * losses[0], losses
