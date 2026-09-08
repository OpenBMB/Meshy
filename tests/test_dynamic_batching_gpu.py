"""GPU equivalence tests for the two dynamic-batching layouts.

Builds two single-rank TitanTrainers on the Qwen3 debug model with identical
weights — ``padded`` + SDPA and ``packed`` + varlen attention — and checks
that per-token log-probs, the PPO loss and the gradient norm agree. Skipped
without CUDA. Run on one card::

    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_dynamic_batching_gpu.py -q
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

SEQ_LEN = 512
LENGTHS = (300, 120, 50, 200, 33, 260)


def _single_rank_env():
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")


def _samples(seed: int = 0):
    from tensordict import TensorDict

    g = torch.Generator().manual_seed(seed)
    out = []
    for L in LENGTHS:
        mask = torch.ones(L)
        mask[: L // 3] = 0
        out.append(TensorDict({
            "tokens": torch.randint(1, 2000, (L,), generator=g),
            "logprobs": -torch.rand(L, generator=g) * 2,
            "mask_assistant": mask,
            "advantage": torch.randn((), generator=g),
        }, batch_size=[]))
    return out


def _build(layout: str, **params):
    from meshy.backend.titan.config import build_forge_config
    from meshy.backend.titan.trainer import TitanTrainer
    from meshy.config import TrainerConfig

    cfg = TrainerConfig(
        model_name="qwen3", model_flavor="debugmodel", seq_len=SEQ_LEN,
        attn_backend="varlen" if layout == "packed" else "sdpa",
        dp_shard_degree=-1, enable_checkpoint=False, compile_model=False,
        dump_folder="/tmp/xrl_dyn_batch_test", lr=0.0, max_norm=1e9,
    )
    return TitanTrainer(
        build_forge_config(cfg), batch_layout=layout, seq_bucket=64,
        timer_enabled=False, **params,
    )


def _copy_weights(src, dst):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, set_model_state_dict,
    )

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_model_state_dict(src.model_parts[0], options=opts)
    set_model_state_dict(dst.model_parts[0], sd, options=opts)


@pytest.fixture(scope="module")
def trainers():
    _single_rank_env()
    torch.manual_seed(0)
    # ``train`` behaviour policy: ratio == 1 at step 0, nothing is clipped and
    # the gradient is a plain -adv * grad(log p) -- a meaningful signal for
    # comparing the layouts (random rollout logprobs would clip every token).
    padded = _build("padded", micro_batch_size=3, mini_batch_size=6,
                    old_logprobs_source="train")
    packed = _build("packed", max_tokens_per_micro=640, mini_batch_size=6,
                    old_logprobs_source="train")
    _copy_weights(padded, packed)
    return padded, packed


def _per_sample_logprobs(trainer, samples):
    """Forward every sample through ``trainer``'s own plan; return {sample: lp[L]}."""
    full = trainer.plan_batch(samples, 1)
    local = full.local_samples(samples, 0)
    out: dict[int, torch.Tensor] = {}
    with torch.no_grad(), trainer.train_context():
        for mini in full.per_rank[0]:
            for micro in mini.micros:
                mb = trainer._build_micro(local, micro, need_rollout_lp=False)
                lp = trainer._forward_batch(mb).float()
                for j, (idx, L) in enumerate(zip(micro.sample_idx, micro.doc_lens)):
                    # Row padding shares the row's doc id; positions tell it apart.
                    sel = (mb.doc_ids == j) & (mb.positions < L)
                    out[full.local_indices[0][idx]] = lp[sel]
    return out


def test_layouts_give_same_logprobs(trainers):
    padded, packed = trainers
    samples = _samples()
    a = _per_sample_logprobs(padded, samples)
    b = _per_sample_logprobs(packed, samples)
    assert a.keys() == b.keys() == set(range(len(samples)))
    for i in a:
        assert a[i].shape == b[i].shape == (LENGTHS[i],)
        # Both kernels run in bf16; compare on the loss-carrying positions.
        L = LENGTHS[i]
        keep = torch.arange(L, device=a[i].device) < L - 1
        torch.testing.assert_close(a[i][keep], b[i][keep], rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("per_token", [False, True])
def test_layouts_give_same_loss_and_grad(trainers, per_token):
    padded, packed = trainers
    samples = _samples(seed=1)
    results = {}
    for name, tr in (("padded", padded), ("packed", packed)):
        tr.calculate_per_token_loss = per_token
        results[name] = tr.train_step(samples)
    a, b = results["padded"], results["packed"]
    assert a["train/n_micro"] >= 2 and b["train/n_micro"] >= 1
    assert b["train/padding_ratio"] < a["train/padding_ratio"]
    assert abs(a["pg_loss"] - b["pg_loss"]) < 2e-2 * max(1.0, abs(a["pg_loss"]))
    assert abs(a["grad_norm"] - b["grad_norm"]) < 3e-2 * max(1.0, a["grad_norm"])
    assert abs(a["ratio_mean"] - b["ratio_mean"]) < 2e-2
    assert abs(a["ratio_mean"] - 1.0) < 1e-3 and a["clip_frac"] == 0.0
    assert a["grad_norm"] > 1e-3


def test_loss_is_a_global_mean_over_samples(trainers):
    """Splitting the same samples over more micro-batches must not change the loss."""
    padded, _ = trainers
    padded.calculate_per_token_loss = False
    samples = _samples(seed=2)
    one = padded.train_step(samples)
    # Force one sample per micro-batch via the row cap.
    from meshy.backend.titan.plan import PlannerConfig
    saved = padded.planner_config
    padded.planner_config = PlannerConfig(
        layout="padded", mini_batch_size=6, seq_len=SEQ_LEN, align=padded.seq_align,
        max_tokens_per_micro=None, micro_batch_size=1,
    )
    try:
        many = padded.train_step(samples)
    finally:
        padded.planner_config = saved
    assert many["train/n_micro"] == len(LENGTHS)
    assert abs(one["pg_loss"] - many["pg_loss"]) < 1e-2 * max(1.0, abs(one["pg_loss"]))
    assert abs(one["grad_norm"] - many["grad_norm"]) < 2e-2 * max(1.0, one["grad_norm"])


def test_filler_micro_contributes_nothing(trainers):
    from meshy.backend.titan.plan import MicroPlan, MiniPlan

    padded, _ = trainers
    full = padded.plan_batch(_samples(seed=3), 1)
    samples = full.local_samples(_samples(seed=3), 0)
    mini = full.per_rank[0][0]
    with_filler = MiniPlan(
        micros=mini.micros + (MicroPlan((), padded.seq_align, ()),),
        n_docs_global=mini.n_docs_global,
        n_tokens_global=mini.n_tokens_global,
        tokens_per_rank=mini.tokens_per_rank,
    )
    ref = padded.train_step(samples, plan=[mini])
    got = padded.train_step(samples, plan=[with_filler])
    assert got["train/n_micro"] == ref["train/n_micro"] + 1
    assert abs(ref["pg_loss"] - got["pg_loss"]) < 1e-6
    assert abs(ref["grad_norm"] - got["grad_norm"]) < 1e-4 * max(1.0, ref["grad_norm"])
