"""Unit tests for dynamic batching: the planner and micro-batch construction.

Everything here runs on CPU without a process group. The CP sharder is
replaced by an identity stub because ``build_micro_batch`` only touches it
through ``shard_seq``.
"""

from __future__ import annotations

import random

import pytest
import torch
from tensordict import TensorDict

from meshy.backend.titan.batch import build_micro_batch
from meshy.backend.titan.plan import (
    MicroPlan,
    PlannerConfig,
    build_plan,
    plan_stats,
    resolve_align,
    round_up,
)
from meshy.config import TrainerConfig, TrainerParamsConfig


class _NoCp:
    enabled = False

    def shard_seq(self, *tensors):
        return tensors


def _cfg(**kw) -> PlannerConfig:
    base = dict(layout="padded", mini_batch_size=4, seq_len=4096, align=64,
                max_tokens_per_micro=None, micro_batch_size=1)
    base.update(kw)
    return PlannerConfig(**base)


def _sample(length: int, prompt: int = 3, adv: float = 1.0, seed: int = 0) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(length)
    mask[:prompt] = 0
    return TensorDict(
        {
            "tokens": torch.randint(1, 1000, (length,), generator=g),
            "logprobs": torch.randn(length, generator=g),
            "mask_assistant": mask,
            "advantage": torch.tensor(adv),
        },
        batch_size=[],
    )


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------

def _check_invariants(plan, lengths, cfg, dp_size):
    n = len(lengths)
    assert plan.dp_size == dp_size
    seen = sorted(i for idx in plan.local_indices for i in idx)
    assert seen == list(range(n)), "every sample assigned exactly once"
    n_mini = len(plan.per_rank[0])
    for minis in plan.per_rank:
        assert len(minis) == n_mini
    for i in range(n_mini):
        counts = {len(plan.per_rank[r][i].micros) for r in range(dp_size)}
        assert len(counts) == 1, "micro counts must match across ranks"
        denom = {(m.n_docs_global, m.n_tokens_global) for m in (plan.per_rank[r][i] for r in range(dp_size))}
        assert len(denom) == 1, "global denominators must match across ranks"
    for r, minis in enumerate(plan.per_rank):
        local = plan.local_indices[r]
        covered = []
        for mini in minis:
            assert len(mini.micros) > 0
            for m in mini.micros:
                assert m.seq_len % cfg.align == 0
                assert m.seq_len <= cfg.seq_len
                for p, L in zip(m.sample_idx, m.doc_lens):
                    assert L == min(lengths[local[p]], cfg.seq_len)
                    covered.append(p)
                if cfg.layout == "padded":
                    assert all(L <= m.seq_len for L in m.doc_lens)
                else:
                    assert m.n_tokens <= m.seq_len
                if cfg.max_tokens_per_micro is not None and len(m.sample_idx) > 1:
                    cost = m.n_rows * m.seq_len if cfg.layout == "padded" else m.seq_len
                    assert cost <= cfg.max_tokens_per_micro
                if cfg.max_tokens_per_micro is None and not m.is_filler:
                    assert len(m.sample_idx) <= cfg.micro_batch_size
        assert sorted(covered) == list(range(len(local)))


@pytest.mark.parametrize("layout", ["padded", "packed"])
@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_plan_invariants(layout, dp_size):
    rng = random.Random(7)
    lengths = [rng.randint(5, 5000) for _ in range(32)]
    loss_tokens = [max(0, L - 4) for L in lengths]
    cfg = _cfg(layout=layout, max_tokens_per_micro=3000, mini_batch_size=4)
    plan = build_plan(lengths, loss_tokens, dp_size, cfg)
    _check_invariants(plan, lengths, cfg, dp_size)
    # Global denominators sum to the batch.
    total_docs = sum(m.n_docs_global for m in plan.per_rank[0])
    assert total_docs == 32
    total_tokens = sum(m.n_tokens_global for m in plan.per_rank[0])
    assert total_tokens == sum(loss_tokens)


def test_plan_is_deterministic():
    lengths = [random.Random(1).randint(1, 3000) for _ in range(16)]
    cfg = _cfg(layout="packed", max_tokens_per_micro=4096)
    a = build_plan(lengths, lengths, 2, cfg)
    b = build_plan(lengths, lengths, 2, cfg)
    assert a == b


def test_dp_partition_balances_tokens_better_than_contiguous_slices():
    # 8 groups of 4 rollouts, group k has length 100 * (k + 1): contiguous
    # slicing gives rank 0 the short groups and rank 3 the long ones.
    lengths = [100 * (k + 1) for k in range(8) for _ in range(4)]
    cfg = _cfg(mini_batch_size=8, micro_batch_size=8)
    plan = build_plan(lengths, lengths, 4, cfg)
    per_rank = [sum(lengths[i] for i in idx) for idx in plan.local_indices]
    contiguous = [sum(lengths[r * 8:(r + 1) * 8]) for r in range(4)]
    assert max(per_rank) - min(per_rank) < max(contiguous) - min(contiguous)
    assert all(len(idx) == 8 for idx in plan.local_indices)


def test_padded_micro_uses_longest_sample_rounded_up():
    lengths = [1000, 130, 70]
    cfg = _cfg(mini_batch_size=3, micro_batch_size=3, align=64)
    plan = build_plan(lengths, lengths, 1, cfg)
    (mini,) = plan.per_rank[0]
    (micro,) = mini.micros
    assert micro.seq_len == round_up(1000, 64)
    assert sorted(micro.doc_lens) == sorted(lengths)


def test_token_budget_padded_groups_by_rows_times_row_len():
    lengths = [900, 800, 100, 90, 80]
    cfg = _cfg(mini_batch_size=5, max_tokens_per_micro=2000, align=64)
    plan = build_plan(lengths, lengths, 1, cfg)
    (mini,) = plan.per_rank[0]
    for m in mini.micros:
        assert m.n_rows * m.seq_len <= 2000
    # 900 -> row_len 960: only two rows fit; 100/90/80 -> row_len 128: all three fit.
    sizes = sorted(len(m.sample_idx) for m in mini.micros)
    assert sizes == [2, 3]


def test_token_budget_packed_fills_up_to_budget():
    lengths = [1500, 1000, 700, 600, 200]
    cfg = _cfg(layout="packed", mini_batch_size=5, max_tokens_per_micro=2048, align=64)
    plan = build_plan(lengths, lengths, 1, cfg)
    (mini,) = plan.per_rank[0]
    for m in mini.micros:
        assert m.seq_len <= 2048 and m.seq_len % 64 == 0
        assert m.n_tokens <= m.seq_len
    assert sum(len(m.sample_idx) for m in mini.micros) == 5
    groups = sorted(sorted(m.doc_lens, reverse=True) for m in mini.micros)
    # First-fit-decreasing: 1500 | 1000 -> 700 joins 1000 (1728 <= 2048) ->
    # 600 fits nowhere (2112 / 2304) -> 200 joins 1500 (1728).
    assert groups == sorted([[1500, 200], [1000, 700], [600]])


def test_single_sample_over_budget_gets_own_micro():
    lengths = [4000, 10]
    cfg = _cfg(layout="packed", mini_batch_size=2, max_tokens_per_micro=1024, align=64)
    plan = build_plan(lengths, lengths, 1, cfg)
    (mini,) = plan.per_rank[0]
    assert len(mini.micros) == 2
    big = next(m for m in mini.micros if 4000 in m.doc_lens)
    assert big.sample_idx == (0,) and big.seq_len == 4032


def test_filler_micros_equalise_counts_across_ranks():
    # Rank with the long sample needs more micros than the other one.
    lengths = [3000, 100, 100, 100]
    cfg = _cfg(layout="packed", mini_batch_size=2, max_tokens_per_micro=1024, align=64)
    plan = build_plan(lengths, lengths, 2, cfg)
    counts = [len(plan.per_rank[r][0].micros) for r in range(2)]
    assert counts[0] == counts[1]
    fillers = [m for r in range(2) for m in plan.per_rank[r][0].micros if m.is_filler]
    assert fillers, "expected at least one filler micro"
    assert all(m.seq_len == 64 and m.doc_lens == () for m in fillers)


def test_truncation_to_seq_len():
    lengths = [10_000, 50]
    cfg = _cfg(mini_batch_size=2, micro_batch_size=2, seq_len=4096, align=64)
    plan = build_plan(lengths, lengths, 1, cfg)
    (mini,) = plan.per_rank[0]
    assert max(L for m in mini.micros for L in m.doc_lens) == 4096


def test_batch_not_divisible_by_dp_raises():
    with pytest.raises(ValueError, match="not divisible"):
        build_plan([1, 2, 3], [1, 2, 3], 2, _cfg())


def test_resolve_align():
    assert resolve_align(2048, 8, 131072) == 2048
    assert resolve_align(2048, 8, 512) == 512      # bucket larger than seq_len
    assert resolve_align(100, 8, 4000) == 200      # lcm(100, 8)
    with pytest.raises(ValueError):
        resolve_align(3000, 8, 4096)               # 4096 % 3000 != 0


def test_planner_config_validation():
    with pytest.raises(ValueError):
        _cfg(layout="packed", max_tokens_per_micro=None)
    with pytest.raises(ValueError):
        _cfg(seq_len=100, align=64)
    with pytest.raises(ValueError):
        _cfg(max_tokens_per_micro=None, micro_batch_size=None)


def test_plan_stats_padding_ratio():
    lengths = [100, 100]
    cfg = _cfg(mini_batch_size=2, micro_batch_size=2, align=128)
    plan = build_plan(lengths, lengths, 1, cfg)
    stats = plan_stats(plan.per_rank[0], "padded")
    assert stats["n_micro"] == 1
    assert stats["tokens_per_micro"] == 200
    assert stats["padding_ratio"] == pytest.approx(1 - 200 / 256)


# ----------------------------------------------------------------------
# Micro-batch construction
# ----------------------------------------------------------------------

def _expected_shift(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[1:], x.new_zeros(1)])


def test_padded_batch_matches_per_sample_shift():
    samples = [_sample(10, seed=1), _sample(6, seed=2, adv=-0.5)]
    micro = MicroPlan((0, 1), 16, (10, 6))
    mb = build_micro_batch(samples, micro, layout="padded", device=torch.device("cpu"),
                           sharder=_NoCp(), need_rollout_lp=True)
    assert mb.input_ids.shape == (2, 16)
    assert mb.n_docs == 2 and mb.n_slots == 3
    for j, (s, L) in enumerate(zip(samples, (10, 6))):
        assert torch.equal(mb.input_ids[j, :L], s["tokens"][:L])
        assert torch.equal(mb.labels[j, :L], _expected_shift(s["tokens"][:L]))
        assert torch.equal(mb.mask[j, :L], _expected_shift(s["mask_assistant"][:L]))
        assert torch.equal(mb.rollout_logprobs[j, :L], _expected_shift(s["logprobs"][:L]))
        assert (mb.input_ids[j, L:] == 0).all() and (mb.mask[j, L:] == 0).all()
        assert (mb.doc_ids[j] == j).all()
    assert torch.equal(mb.positions[0], torch.arange(16, dtype=torch.int32))
    assert torch.allclose(mb.advantages, torch.tensor([1.0, -0.5]))
    assert mb.attention_masks is None


def test_packed_batch_layout_and_metadata():
    samples = [_sample(10, seed=1), _sample(6, seed=2, adv=2.0)]
    micro = MicroPlan((0, 1), 32, (10, 6))
    mb = build_micro_batch(samples, micro, layout="packed", device=torch.device("cpu"),
                           sharder=_NoCp(), need_rollout_lp=True)
    assert mb.input_ids.shape == (1, 32)
    row = mb.input_ids[0]
    assert torch.equal(row[:10], samples[0]["tokens"])
    assert torch.equal(row[10:16], samples[1]["tokens"])
    assert (row[16:] == 0).all()
    # Per-document shift: label at the last token of each doc is 0, mask 0.
    assert torch.equal(mb.labels[0, :10], _expected_shift(samples[0]["tokens"]))
    assert torch.equal(mb.labels[0, 10:16], _expected_shift(samples[1]["tokens"]))
    assert mb.mask[0, 9] == 0 and mb.mask[0, 15] == 0
    assert torch.equal(mb.rollout_logprobs[0, 10:16], _expected_shift(samples[1]["logprobs"]))
    # Positions restart per doc, including the padding doc.
    assert torch.equal(mb.positions[0, :10], torch.arange(10, dtype=torch.int32))
    assert torch.equal(mb.positions[0, 10:16], torch.arange(6, dtype=torch.int32))
    assert torch.equal(mb.positions[0, 16:], torch.arange(16, dtype=torch.int32))
    # doc ids: 0, 1, then padding slot n_docs.
    assert (mb.doc_ids[0, :10] == 0).all() and (mb.doc_ids[0, 10:16] == 1).all()
    assert (mb.doc_ids[0, 16:] == 2).all()
    meta = mb.attention_masks
    assert torch.equal(meta.cu_seq_q, torch.tensor([0, 10, 16, 32], dtype=torch.int32))
    assert meta.cu_seq_k is meta.cu_seq_q or torch.equal(meta.cu_seq_k, meta.cu_seq_q)
    assert meta.max_q == 16 and meta.max_k == 16
    assert torch.allclose(mb.advantages, torch.tensor([1.0, 2.0]))


def test_packed_batch_without_padding_has_no_pad_doc():
    samples = [_sample(8, seed=3)]
    micro = MicroPlan((0,), 8, (8,))
    mb = build_micro_batch(samples, micro, layout="packed", device=torch.device("cpu"),
                           sharder=_NoCp(), need_rollout_lp=False)
    assert torch.equal(mb.attention_masks.cu_seq_q, torch.tensor([0, 8], dtype=torch.int32))
    assert mb.rollout_logprobs is None


@pytest.mark.parametrize("layout", ["padded", "packed"])
def test_filler_micro_is_all_padding(layout):
    micro = MicroPlan((), 64, ())
    mb = build_micro_batch([], micro, layout=layout, device=torch.device("cpu"),
                           sharder=_NoCp(), need_rollout_lp=True)
    assert mb.n_docs == 0
    assert mb.input_ids.shape == (1, 64)
    assert (mb.mask == 0).all()
    assert (mb.doc_ids == 0).all()   # slot 0 == n_docs == padding
    assert mb.advantages.numel() == 0


def test_sample_truncation_in_micro():
    samples = [_sample(20, seed=4)]
    micro = MicroPlan((0,), 16, (12,))
    mb = build_micro_batch(samples, micro, layout="padded", device=torch.device("cpu"),
                           sharder=_NoCp(), need_rollout_lp=False)
    assert torch.equal(mb.input_ids[0, :12], samples[0]["tokens"][:12])
    assert mb.labels[0, 11] == 0 and mb.mask[0, 11] == 0
    assert int(mb.lengths[0]) == 12


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------

def test_trainer_params_config_validation():
    TrainerParamsConfig(batch_layout="padded")
    TrainerParamsConfig(batch_layout="packed", max_tokens_per_micro=8192)
    with pytest.raises(ValueError):
        TrainerParamsConfig(batch_layout="packed")
    with pytest.raises(ValueError):
        TrainerParamsConfig(max_tokens_per_micro=0)
    with pytest.raises(ValueError):
        TrainerParamsConfig(seq_bucket=0)
    with pytest.raises(ValueError):
        TrainerConfig(attn_backend="flash")


def test_forge_config_threads_attn_backend():
    from meshy.backend.titan.config import build_forge_config
    from meshy.backend.titan.trainer import _attn_backend_of

    for name, flavor in (("qwen3", "0.6B"), ("qwen2_5", "0.5B"), ("minicpm5", "2B")):
        for backend in ("sdpa", "varlen"):
            cfg = build_forge_config(TrainerConfig(
                model_name=name, model_flavor=flavor, attn_backend=backend,
            ))
            assert _attn_backend_of(cfg.model_spec.model) == backend
