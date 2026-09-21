"""The global checkpoint's derivation rules, exercised without any GPU.

A Meshy run has no driver, so "which step do we resume from" is not recorded
anywhere -- it is *derived*, independently, by every process, from a directory
listing plus the recipe. These tests pin the three rules that makes safe:

* a shard counts only once its ``.complete`` marker lands, so a crash
  mid-write costs one interval rather than the run;
* the resume point is the newest step **every** participant reached, not the
  newest step any single role happens to own;
* who participates depends on the step, because the actor trainer does not
  run at all during the critic's cold start.
"""

from __future__ import annotations

import os

import pytest

from meshy.config import GlobalCheckpointConfig
from meshy.service.base import GPU, ServiceGroup
from meshy.service.checkpoint import StateStore, resolve_resume_step
from meshy.service.topology import build_topology


# ── GlobalCheckpointConfig ───────────────────────────────────────────────
def test_due_only_on_positive_multiples_of_the_interval():
    policy = GlobalCheckpointConfig(enabled=True, interval=10, keep_latest_k=2)
    assert not policy.due(0)  # step 0 is "nothing has happened yet"
    assert not policy.due(9)
    assert policy.due(10)
    assert policy.due(20)


def test_disabled_policy_is_never_due():
    assert not GlobalCheckpointConfig(enabled=False, interval=1).due(7)


def test_keep_latest_k_of_one_is_rejected():
    # The newest retained step may be one a slower peer has not written yet,
    # so keeping a single step can leave the intersection empty.
    with pytest.raises(ValueError, match="keep_latest_k"):
        GlobalCheckpointConfig(keep_latest_k=1)


def test_coerce_accepts_the_asdict_form_the_rollout_receives():
    policy = GlobalCheckpointConfig(enabled=True, interval=4, keep_latest_k=2)
    assert GlobalCheckpointConfig.coerce({"enabled": True, "interval": 4, "keep_latest_k": 2}) == policy
    assert GlobalCheckpointConfig.coerce(policy) is policy
    assert GlobalCheckpointConfig.coerce(None) == GlobalCheckpointConfig()


# ── StateStore ───────────────────────────────────────────────────────────
def _commit(store: StateStore, service: str, step: int, **payload) -> None:
    store.write_meta(service, step, payload or {"step": step})
    store.mark_complete(service, step)


def test_meta_without_its_marker_is_not_a_shard(tmp_path):
    store = StateStore(str(tmp_path))
    store.write_meta("critic-0", 10, {"train_steps": 10})
    assert store.complete_steps("critic-0") == []
    assert not store.is_complete("critic-0", 10)
    store.mark_complete("critic-0", 10)
    assert store.complete_steps("critic-0") == [10]


def test_payload_round_trips(tmp_path):
    store = StateStore(str(tmp_path))
    _commit(store, "rollout-0", 20, epoch=0, dataset_index=960)
    assert store.payload("rollout-0", 20) == {"epoch": 0, "dataset_index": 960}
    assert store.payload("rollout-0", 30) == {}


def test_unknown_state_dir_reads_as_empty(tmp_path):
    store = StateStore(str(tmp_path / "never-created"))
    assert store.known_services() == []
    assert store.complete_steps("critic-0") == []
    assert not store.has_any_shard()


def test_resume_point_is_the_intersection_not_the_newest_shard(tmp_path):
    store = StateStore(str(tmp_path))
    for step in (10, 20, 30):
        _commit(store, "critic-0", step)
        _commit(store, "rollout-0", step)
    # The trainer crashed after 20, so 30 is not a run-wide checkpoint even
    # though two of the three roles reached it.
    for step in (10, 20):
        _commit(store, "actor_train-0", step)

    participants = lambda step: ("actor_train-0", "critic-0", "rollout-0")  # noqa: E731
    assert resolve_resume_step(store, participants) == 20


def test_resume_point_skips_a_step_whose_marker_is_missing(tmp_path):
    store = StateStore(str(tmp_path))
    for service in ("critic-0", "rollout-0"):
        _commit(store, service, 10)
        _commit(store, service, 20)
    # A torn write: the heavy shard and its meta exist, the commit does not.
    store.write_meta("critic-0", 30, {})
    _commit(store, "rollout-0", 30)

    participants = lambda step: ("critic-0", "rollout-0")  # noqa: E731
    assert resolve_resume_step(store, participants) == 20


def test_no_common_step_means_no_resume(tmp_path):
    store = StateStore(str(tmp_path))
    _commit(store, "critic-0", 10)
    _commit(store, "rollout-0", 20)
    participants = lambda step: ("critic-0", "rollout-0")  # noqa: E731
    assert resolve_resume_step(store, participants) == 0


def test_prune_keeps_the_newest_k_and_removes_shard_and_sidecars(tmp_path):
    store = StateStore(str(tmp_path))
    for step in (10, 20, 30, 40):
        os.makedirs(store.shard_dir("critic-0", step), exist_ok=True)
        _commit(store, "critic-0", step)

    assert store.prune("critic-0", 2) == [10, 20]
    assert store.complete_steps("critic-0") == [30, 40]
    assert not os.path.exists(store.shard_dir("critic-0", 10))
    assert not os.path.exists(store.meta_path("critic-0", 10))
    assert os.path.isdir(store.shard_dir("critic-0", 40))


def test_drop_newer_than_clears_the_crash_tail(tmp_path):
    store = StateStore(str(tmp_path))
    for step in (10, 20):
        _commit(store, "critic-0", step)
        _commit(store, "rollout-0", step)
        _commit(store, "actor_train-0", step)
    # The critic runs before the trainer, so a crash mid-window leaves it one
    # shard ahead. That shard can never be a resume point and the resumed run
    # rewrites the same step number.
    _commit(store, "critic-0", 30)
    _commit(store, "rollout-0", 30)

    assert store.drop_newer_than(20) == {"critic-0": [30], "rollout-0": [30]}
    assert store.complete_steps("critic-0") == [10, 20]
    assert store.complete_steps("rollout-0") == [10, 20]
    assert store.complete_steps("actor_train-0") == [10, 20]


def test_drop_newer_than_leaves_a_clean_tree_alone(tmp_path):
    store = StateStore(str(tmp_path))
    for step in (10, 20):
        _commit(store, "critic-0", step)
    assert store.drop_newer_than(20) == {}
    assert store.complete_steps("critic-0") == [10, 20]


def test_prune_is_a_no_op_when_keeping_everything(tmp_path):
    store = StateStore(str(tmp_path))
    for step in (10, 20, 30):
        _commit(store, "critic-0", step)
    assert store.prune("critic-0", 0) == []
    assert store.complete_steps("critic-0") == [10, 20, 30]


# ── participants: the cold-start rule ────────────────────────────────────
def _topology(cold_start_windows: int, *, policy: GlobalCheckpointConfig, cards: int = 4):
    from meshy.config import (
        CriticServiceConfig,
        InferenceServiceConfig,
        RolloutServiceConfig,
        TrainerConfig,
        TrainingServiceConfig,
    )
    from meshy.service.colocation import ColocationRing, SchedulingMode

    trainer_config = TrainerConfig(model_name="qwen3", model_flavor="1.7B")
    groups = [
        ServiceGroup(
            id="actor_infer",
            n_replicas=cards,
            n_gpus_per_replica=1,
            config=InferenceServiceConfig(model_path="m"),
        ),
        ServiceGroup(
            id="actor_train",
            n_gpus_per_replica=cards,
            colocate_with="actor_infer",
            config=TrainingServiceConfig(
                model_path="m",
                trainer_config=trainer_config,
                batch_size=8,
                critic_cold_start_windows=cold_start_windows,
                checkpoint=policy,
            ),
        ),
        ServiceGroup(
            id="critic",
            n_gpus_per_replica=cards,
            config=CriticServiceConfig(
                model_path="m",
                trainer_config=trainer_config,
                score_batch_size=8,
                publish_mode="advantage",
                cold_start_windows=cold_start_windows,
                checkpoint=policy,
            ),
        ),
        ServiceGroup(
            id="rollout",
            n_gpus_per_replica=0,
            config=RolloutServiceConfig(
                model_path="m",
                dataset="meshy.dataset.gsm8k:GSM8K",
                checkpoint=policy,
            ),
        ),
    ]
    gpus = [
        GPU(host="h", global_rank=i, node_rank=0, local_rank=i) for i in range(cards * 2)
    ]
    colocations = [
        ColocationRing(
            group_id="actor_card",
            ring=(
                ("actor_infer", SchedulingMode.FALLBACK),
                ("actor_train", SchedulingMode.ON_DEMAND),
            ),
        )
    ]
    return build_topology(groups, gpus, colocations)


def test_trainer_is_not_a_participant_during_the_critic_cold_start():
    policy = GlobalCheckpointConfig(enabled=True, interval=2, keep_latest_k=2)
    topology = _topology(20, policy=policy)
    # Inside the cold start the trainer's TQ fetch never matches, so it takes
    # no step and writes no shard. Requiring one would make the whole phase
    # unresumable -- and that phase is pure critic training with nothing else
    # to show for it.
    assert topology.checkpoint_participants(10) == ("critic-0", "rollout-0")
    assert topology.checkpoint_participants(20) == ("critic-0", "rollout-0")
    assert topology.checkpoint_participants(21) == (
        "actor_train-0",
        "critic-0",
        "rollout-0",
    )


def test_an_endless_cold_start_never_expects_a_trainer_shard():
    policy = GlobalCheckpointConfig(enabled=True, interval=2, keep_latest_k=2)
    topology = _topology(-1, policy=policy)
    assert topology.checkpoint_participants(1000) == ("critic-0", "rollout-0")


def test_a_cold_start_resume_needs_only_the_critic_and_the_rollout(tmp_path):
    policy = GlobalCheckpointConfig(enabled=True, interval=10, keep_latest_k=2)
    topology = _topology(20, policy=policy)
    store = StateStore(str(tmp_path))
    for step in (10, 20):
        _commit(store, "critic-0", step)
        _commit(store, "rollout-0", step)
    assert resolve_resume_step(store, topology.checkpoint_participants) == 20


def test_disabled_checkpointing_leaves_no_participants_and_no_policy():
    topology = _topology(20, policy=GlobalCheckpointConfig(enabled=False))
    assert topology.checkpoint_config() is None
    assert topology.checkpoint_participants(50) == ()


# ── recipe validation ────────────────────────────────────────────────────
def test_services_on_different_checkpoint_policies_are_rejected():
    from meshy.config import (
        InferenceServiceConfig,
        RolloutServiceConfig,
        TrainerConfig,
        TrainingServiceConfig,
    )

    groups = [
        ServiceGroup(
            id="actor_infer", config=InferenceServiceConfig(model_path="m")
        ),
        ServiceGroup(
            id="actor_train",
            colocate_with="actor_infer",
            config=TrainingServiceConfig(
                model_path="m",
                trainer_config=TrainerConfig(),
                batch_size=8,
                checkpoint=GlobalCheckpointConfig(enabled=True, interval=10, keep_latest_k=2),
            ),
        ),
        ServiceGroup(
            id="rollout",
            n_gpus_per_replica=0,
            config=RolloutServiceConfig(
                model_path="m",
                dataset="meshy.dataset.gsm8k:GSM8K",
                # A different interval: the two would only ever intersect at
                # their LCM, and the failure is silent without this check.
                checkpoint=GlobalCheckpointConfig(enabled=True, interval=7, keep_latest_k=2),
            ),
        ),
    ]
    gpus = [GPU(host="h", global_rank=0, node_rank=0, local_rank=0)]
    with pytest.raises(ValueError, match="share one GlobalCheckpointConfig"):
        build_topology(groups, gpus)


def test_a_role_left_out_of_checkpointing_is_rejected():
    from meshy.config import (
        InferenceServiceConfig,
        RolloutServiceConfig,
        TrainerConfig,
        TrainingServiceConfig,
    )

    policy = GlobalCheckpointConfig(enabled=True, interval=10, keep_latest_k=2)
    groups = [
        ServiceGroup(id="actor_infer", config=InferenceServiceConfig(model_path="m")),
        ServiceGroup(
            id="actor_train",
            colocate_with="actor_infer",
            config=TrainingServiceConfig(
                model_path="m",
                trainer_config=TrainerConfig(),
                batch_size=8,
                checkpoint=policy,
            ),
        ),
        # The rollout would silently rewind to the top of the dataset while
        # the actor resumed mid-run.
        ServiceGroup(
            id="rollout",
            n_gpus_per_replica=0,
            config=RolloutServiceConfig(
                model_path="m", dataset="meshy.dataset.gsm8k:GSM8K"
            ),
        ),
    ]
    gpus = [GPU(host="h", global_rank=0, node_rank=0, local_rank=0)]
    with pytest.raises(ValueError, match="'rollout' service"):
        build_topology(groups, gpus)
