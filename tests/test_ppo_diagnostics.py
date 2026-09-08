"""PPO loss diagnostics: ``ppo_kl`` / ``train_rollout_kl`` / ``ess_ratio`` / entropy.

Runs ``TitanTrainer._ppo_clip_loss`` unbound on a stub, so no model, GPU or
process group is needed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch


class _NoCp:
    enabled = False

    def all_reduce_sum(self, value):
        return value


def _loss_stub(**overrides):
    from meshy.backend.titan.trainer import TitanTrainer as Backend

    stub = SimpleNamespace(
        ppo_clip_eps_low=0.2, ppo_clip_eps_high=0.28, use_tis=True,
        tis_ratio_min=0.5, tis_ratio_max=5.0, calculate_per_token_loss=True,
        old_logprobs_source="rollout", sharder=_NoCp(), parallel_dims=SimpleNamespace(cp=1),
    )
    for k, v in overrides.items():
        setattr(stub, k, v)
    return Backend, stub


def _batch(rows=2, S=8):
    from meshy.backend.titan.batch import Batch

    mask = torch.ones(rows, S)
    mask[:, -1] = 0
    doc_ids = torch.arange(rows).unsqueeze(1).expand(rows, S).clone()
    return Batch(
        input_ids=torch.zeros(rows, S, dtype=torch.long), labels=torch.zeros(rows, S, dtype=torch.long),
        positions=torch.zeros(rows, S, dtype=torch.int32), mask=mask, doc_ids=doc_ids,
        advantages=torch.tensor([1.0, -1.0]), rollout_logprobs=None,
        lengths=torch.tensor([S, S], dtype=torch.int32), n_docs=rows,
    )


def _reduce(sums):
    from meshy.backend.titan.trainer import TitanTrainer as Backend

    return Backend._reduce_mini_metrics(
        SimpleNamespace(parallel_dims=SimpleNamespace(get_optional_mesh=lambda name: None)), sums
    )


def test_ppo_loss_reports_kl_and_ess():
    from meshy.backend.titan.plan import MiniPlan

    Backend, stub = _loss_stub()
    mb = _batch()
    mini = MiniPlan(micros=(), n_docs_global=2, n_tokens_global=int(mb.mask.sum()), tokens_per_rank=(16,))
    old = -torch.ones(2, 8)
    # new = old + delta: ratio = exp(delta), constant per token
    delta = torch.full((2, 8), 0.1)
    new = (old + delta).requires_grad_()
    mb.rollout_logprobs = old.clone()
    loss, sums = Backend._ppo_clip_loss(stub, new, old, mb, mini, entropy=torch.full((2, 8), 2.0))
    red = _reduce(sums)
    assert red["ppo_kl"] == pytest.approx(-0.1)          # old - new
    assert red["log_ratio_abs_mean"] == pytest.approx(0.1)
    assert red["ratio_mean"] == pytest.approx(math.exp(0.1))
    assert red["ess_ratio"] == pytest.approx(1.0)        # constant weights -> full ESS
    assert red["entropy"] == pytest.approx(2.0)
    # k3 between rollout (= old) and new: e^{-d} + d - 1
    assert red["train_rollout_kl"] == pytest.approx(math.exp(-0.1) + 0.1 - 1, rel=1e-4)
    assert red["train_rollout_logdiff_abs"] == pytest.approx(0.1)
    assert sums["token_count"].item() == float(mb.mask.sum())
    loss.backward()  # the diagnostics must not break autograd


def test_ppo_loss_ess_drops_with_uneven_weights():
    from meshy.backend.titan.plan import MiniPlan

    Backend, stub = _loss_stub()
    mb = _batch()
    mini = MiniPlan(micros=(), n_docs_global=2, n_tokens_global=int(mb.mask.sum()), tokens_per_rank=(16,))
    old = torch.zeros(2, 8)
    new = torch.zeros(2, 8)
    new[:, 0] = 1.5  # one heavy token per sequence
    _, sums = Backend._ppo_clip_loss(stub, new, old, mb, mini)
    red = _reduce(sums)
    assert 0.0 < red["ess_ratio"] < 1.0
    assert "train_rollout_kl" not in red  # no rollout log-probs on this batch
