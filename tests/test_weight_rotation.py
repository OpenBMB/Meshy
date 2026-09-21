"""Rotation of the exported HF weight versions (``weights/{service}/v{N}``).

One full model is written per training step purely so a colocated SGLang can
reload the policy, and nothing reads an old version once the next one lands.
Left unrotated that is terabytes over a few hundred steps -- long runs have
died of exactly that. The one thing rotation must not do is delete a version a
retained global checkpoint still names, because a resumed inference server
boots from that path.
"""

from __future__ import annotations

import os

from meshy.config import GlobalCheckpointConfig
from meshy.engine.titan import TitanEngine
from meshy.service.checkpoint import StateStore
from meshy.service.runtime import RuntimeDir


class _FakeWeightPruner:
    """``TitanEngine``'s pruning surface, without a process group or a model."""

    _prune_weights = TitanEngine._prune_weights
    _protected_weight_versions = TitanEngine._protected_weight_versions
    _weights_path = TitanEngine._weights_path
    store = TitanEngine.store

    def __init__(self, runtime: RuntimeDir, keep: int, *, name="actor_train-0"):
        self.runtime = runtime
        self.name = name
        self.weights_keep_latest_k = keep
        self.is_master = True
        self._store = None


def _runtime(tmp_path) -> RuntimeDir:
    return RuntimeDir(
        str(tmp_path / "run"),
        checkpoint_dir=str(tmp_path / "weights"),
        state_dir=str(tmp_path / "state"),
    )


def _export(runtime: RuntimeDir, name: str, version: int) -> str:
    path = runtime.checkpoint_path(name, version)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.safetensors"), "w") as handle:
        handle.write("x")
    return path


def _versions(runtime: RuntimeDir, name: str) -> list[int]:
    return sorted(
        int(entry[1:])
        for entry in os.listdir(runtime.weights_dir(name))
        if entry.startswith("v")
    )


def test_keeps_the_newest_k(tmp_path):
    runtime = _runtime(tmp_path)
    for version in range(1, 9):
        _export(runtime, "actor_train-0", version)
    _FakeWeightPruner(runtime, keep=3)._prune_weights(8)
    assert _versions(runtime, "actor_train-0") == [6, 7, 8]


def test_never_prunes_a_version_a_retained_checkpoint_points_at(tmp_path):
    runtime = _runtime(tmp_path)
    store = StateStore(runtime.state_dir)
    for version in range(1, 9):
        _export(runtime, "actor_train-0", version)
    # Two retained global checkpoints, naming v2 and v5. A resumed inference
    # server boots from one of those paths, so deleting them would leave the
    # DCP shard intact and the checkpoint unusable.
    for global_step, version in ((12, 2), (15, 5)):
        store.write_meta(
            "actor_train-0",
            global_step,
            {"weight_version": version, "weights_path": runtime.checkpoint_path("actor_train-0", version)},
        )
        store.mark_complete("actor_train-0", global_step)

    _FakeWeightPruner(runtime, keep=2)._prune_weights(8)
    assert _versions(runtime, "actor_train-0") == [2, 5, 7, 8]


def test_the_version_just_written_survives_a_tiny_keep(tmp_path):
    runtime = _runtime(tmp_path)
    for version in (1, 2, 3):
        _export(runtime, "actor_train-0", version)
    # Guard against a config that would delete the weights the ring is about
    # to hand to SGLang in this very step.
    pruner = _FakeWeightPruner(runtime, keep=1)
    pruner._prune_weights(3)
    assert 3 in _versions(runtime, "actor_train-0")


def test_rotation_off_keeps_everything(tmp_path):
    runtime = _runtime(tmp_path)
    for version in range(1, 6):
        _export(runtime, "actor_train-0", version)
    _FakeWeightPruner(runtime, keep=0)._prune_weights(5)
    assert _versions(runtime, "actor_train-0") == [1, 2, 3, 4, 5]


def test_non_master_ranks_do_not_touch_the_disk(tmp_path):
    runtime = _runtime(tmp_path)
    for version in range(1, 6):
        _export(runtime, "actor_train-0", version)
    pruner = _FakeWeightPruner(runtime, keep=2)
    pruner.is_master = False
    pruner._prune_weights(5)
    assert _versions(runtime, "actor_train-0") == [1, 2, 3, 4, 5]


def test_pruning_before_any_export_is_a_no_op(tmp_path):
    runtime = _runtime(tmp_path)
    _FakeWeightPruner(runtime, keep=2)._prune_weights(0)
