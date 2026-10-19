"""``train_step`` snapshots the behaviour policy before the first optimizer step.

With ``old_logprobs_source="train"`` every mini-batch's old log-probs must come
from the weights the rollout was generated with. Taking them inside each
mini-batch used weights already moved by the earlier mini-batches, which pinned
``ratio`` at 1 and switched the PPO clip / TIS off. Runs ``train_step`` unbound
on a stub, so no model, GPU or process group is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _run(source: str, needs: bool = True, n_mini: int = 3):
    from meshy.backend.titan import trainer as T

    state = {"version": 0, "snapshots": [], "seen": []}

    def compute_old(samples, mini):
        # tag the snapshot with the weight version it was taken at
        state["snapshots"].append(state["version"])
        return [torch.tensor(float(state["version"]))]

    def run_mini(samples, mini, timer, old_lps=None):
        state["seen"].append(None if old_lps is None else float(old_lps[0]))
        state["version"] += 1  # this mini-batch's optimizer.step()
        return {}, torch.tensor(0.0)

    stub = SimpleNamespace(
        timer_enabled=False, enable_gae=False, old_logprobs_source=source,
        needs_behaviour_logprobs=needs, batch_layout="padded", step=0,
        lr_schedulers=SimpleNamespace(step=lambda: None),
        _compute_old_logprobs_train=compute_old, _run_mini_batch=run_mini,
        _aggregate_metrics=lambda metrics, norms, timer: {},
    )
    plan = [SimpleNamespace(micros=[0]) for _ in range(n_mini)]
    mp = pytest.MonkeyPatch()
    mp.setattr(T, "plan_stats", lambda plan, layout: {})
    try:
        T.TitanTrainer.train_step(stub, [], plan=plan)
    finally:
        mp.undo()
    return state


def test_train_source_snapshots_every_mini_batch_before_any_step():
    s = _run("train")
    # one snapshot per mini-batch, all taken at version 0 (before any step)
    assert s["snapshots"] == [0, 0, 0]
    # every mini-batch trains against the version-0 behaviour policy, so the
    # 2nd and 3rd mini-batches see a ratio != 1 once the weights have moved
    assert s["seen"] == [0.0, 0.0, 0.0]
    assert s["version"] == 3


def test_rollout_source_takes_no_snapshot():
    s = _run("rollout")
    assert s["snapshots"] == []
    assert s["seen"] == [None, None, None]


def test_subclass_without_ratio_skips_the_snapshot():
    s = _run("train", needs=False)
    assert s["snapshots"] == []
    assert s["seen"] == [None, None, None]


def test_opd_trainer_opts_out():
    from meshy.backend.titan.opd import StudentTopKTrainer
    from meshy.backend.titan.trainer import TitanTrainer

    assert TitanTrainer.needs_behaviour_logprobs is True
    assert StudentTopKTrainer.needs_behaviour_logprobs is False
