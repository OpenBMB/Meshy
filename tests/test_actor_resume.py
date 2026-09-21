"""The actor engine's counter restore, including the cold-start special case.

``TitanTrainer.step`` comes back through DCP, but the engine's own
``step_index`` / ``weight_version`` / ``last_weights_path`` are in no state
dict and are re-seeded from the shard's sidecar. Two cases have to behave
differently, and getting them confused is silent: a resume *after* the critic's
cold start must find a shard, while a resume *inside* it must not expect one --
the actor has taken no step at all there.
"""

from __future__ import annotations

import os

import pytest

from meshy.config import GlobalCheckpointConfig
from meshy.engine.titan import TitanEngine
from meshy.service.checkpoint import StateStore
from meshy.service.runtime import RuntimeDir


class _StubTrainer:
    def __init__(self, step: int) -> None:
        self.step = step


class _FakeActor:
    """``TitanEngine``'s restore surface, without a process group or a model."""

    _restore_counters = TitanEngine._restore_counters
    store = TitanEngine.store
    global_step = TitanEngine.global_step

    def __init__(self, runtime: RuntimeDir, *, resume_step: int, cold_start: int, trainer_step: int):
        self.runtime = runtime
        self.name = "actor_train-0"
        self.resume_step = resume_step
        self.gate_step_offset = cold_start
        self.is_master = True
        self.trainer = _StubTrainer(trainer_step)
        self.step_index = 0
        self.weight_version = 0
        self.last_weights_path = None
        self._store = None


def _runtime(tmp_path) -> RuntimeDir:
    return RuntimeDir(
        str(tmp_path / "run"),
        checkpoint_dir=str(tmp_path / "weights"),
        state_dir=str(tmp_path / "state"),
    )


def _commit_actor(runtime: RuntimeDir, global_step: int, *, step_index: int, trainer_step: int):
    store = StateStore(runtime.state_dir)
    store.write_meta(
        "actor_train-0",
        global_step,
        {
            "step_index": step_index,
            "weight_version": step_index,
            "weights_path": runtime.checkpoint_path("actor_train-0", step_index),
            "trainer_step": trainer_step,
        },
    )
    store.mark_complete("actor_train-0", global_step)


def test_counters_come_from_the_sidecar_not_from_trainer_step(tmp_path):
    runtime = _runtime(tmp_path)
    # cold start 20, so global step 35 is the actor's step 15.
    _commit_actor(runtime, 35, step_index=15, trainer_step=15)
    actor = _FakeActor(runtime, resume_step=35, cold_start=20, trainer_step=15)
    actor._restore_counters()

    assert actor.step_index == 15
    assert actor.weight_version == 15
    assert actor.last_weights_path.endswith("/actor_train-0/v15")
    # The engine's counter and the run's clock differ by the cold start; the
    # gate numbering downstream depends on exactly this.
    assert actor.global_step == 35


def test_a_cold_start_resume_expects_no_actor_shard(tmp_path):
    runtime = _runtime(tmp_path)
    # Global step 10 with a 20-window cold start: the critic and the rollout
    # have shards, the actor has never stepped.
    actor = _FakeActor(runtime, resume_step=10, cold_start=20, trainer_step=0)
    actor._restore_counters()
    assert (actor.step_index, actor.weight_version, actor.last_weights_path) == (0, 0, None)


def test_a_missing_shard_past_the_cold_start_is_an_error(tmp_path):
    runtime = _runtime(tmp_path)
    actor = _FakeActor(runtime, resume_step=35, cold_start=20, trainer_step=0)
    with pytest.raises(FileNotFoundError, match="shard is incomplete"):
        actor._restore_counters()


def test_a_sidecar_from_another_run_is_rejected(tmp_path):
    runtime = _runtime(tmp_path)
    _commit_actor(runtime, 35, step_index=15, trainer_step=15)
    # DCP restored a different step than the sidecar records: the checkpoint
    # directory and its metadata are not from the same run, and proceeding
    # would export weight versions over a previous run's.
    actor = _FakeActor(runtime, resume_step=35, cold_start=20, trainer_step=9)
    with pytest.raises(RuntimeError, match="not from the same run"):
        actor._restore_counters()


def test_a_fresh_run_restores_nothing(tmp_path):
    runtime = _runtime(tmp_path)
    actor = _FakeActor(runtime, resume_step=0, cold_start=20, trainer_step=0)
    actor._restore_counters()
    assert actor.step_index == 0


def test_global_step_has_no_offset_without_a_critic(tmp_path):
    runtime = _runtime(tmp_path)
    _commit_actor(runtime, 40, step_index=40, trainer_step=40)
    actor = _FakeActor(runtime, resume_step=40, cold_start=0, trainer_step=40)
    actor._restore_counters()
    assert actor.step_index == 40
    assert actor.global_step == 40


def test_inference_boots_from_the_resumed_weights_not_the_base_model(tmp_path, monkeypatch):
    # The genesis GPU grant carries no payload_ref, so without this the
    # inference server would serve the pre-training policy for a whole window
    # while the trainer believed it was generating on v15.
    from meshy.service.checkpoint import RESUME_ENV, RESUME_STEP_ENV
    from meshy.service.inference import _boot_model_path

    monkeypatch.setenv(RESUME_ENV, "1")
    monkeypatch.setenv(RESUME_STEP_ENV, "35")
    runtime = _runtime(tmp_path)
    _commit_actor(runtime, 35, step_index=15, trainer_step=15)
    os.makedirs(runtime.checkpoint_path("actor_train-0", 15), exist_ok=True)

    topology = _FakeTopology()
    assert _boot_model_path("openbmb/Base", topology, runtime) == runtime.checkpoint_path(
        "actor_train-0", 15
    )


def test_inference_keeps_the_base_model_during_a_cold_start_resume(tmp_path, monkeypatch):
    # The actor has not stepped, so the base model *is* the current policy.
    from meshy.service.checkpoint import RESUME_ENV, RESUME_STEP_ENV
    from meshy.service.inference import _boot_model_path

    monkeypatch.setenv(RESUME_ENV, "1")
    monkeypatch.setenv(RESUME_STEP_ENV, "10")
    runtime = _runtime(tmp_path)
    assert _boot_model_path("openbmb/Base", _FakeTopology(), runtime) == "openbmb/Base"


def test_inference_keeps_the_base_model_on_a_fresh_run(tmp_path, monkeypatch):
    from meshy.service.checkpoint import RESUME_ENV
    from meshy.service.inference import _boot_model_path

    monkeypatch.delenv(RESUME_ENV, raising=False)
    runtime = _runtime(tmp_path)
    _commit_actor(runtime, 35, step_index=15, trainer_step=15)
    assert _boot_model_path("openbmb/Base", _FakeTopology(), runtime) == "openbmb/Base"


class _FakeTopology:
    """Just the two hooks ``_boot_model_path`` reaches for."""

    def training_services(self):
        return [type("Info", (), {"name": "actor_train-0"})()]

    def checkpoint_participants(self, step):
        return ("actor_train-0", "critic-0", "rollout-0")


def test_checkpoint_boundaries_are_measured_on_the_global_clock():
    # An interval of 10 with a 20-window cold start must fire at global 30,
    # i.e. the actor's step 10 -- not at the actor's own multiples of 10
    # offset by the cold start.
    policy = GlobalCheckpointConfig(enabled=True, interval=10, keep_latest_k=2)
    assert policy.due(30)
    assert not policy.due(25)
