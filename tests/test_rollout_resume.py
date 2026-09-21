"""The rollout's half of a resume: the data cursor and the pacing counters.

TransferQueue is RAM-only, so a resumed run starts with an empty ``gen_gate``
partition and no in-flight samples. The rollout is therefore the one role that
has to reconstruct *where in the data it was*, and it does so from a cursor it
snapshotted when it consumed a gate. Two properties matter and are pinned here:

* the resumed run does not re-generate prompts it already handed out (groups
  that were still generating are dropped, not replayed);
* ``gates_seen`` comes back one *below* the snapshot so the trainer's
  resume-genesis pulse is consumed as gate ``N`` rather than rewinding the
  stream to 0.
"""

from __future__ import annotations

import pytest

from meshy.config import GlobalCheckpointConfig
from meshy.service.checkpoint import StateStore
from meshy.worker.rollout import _require_seekable, _seek_dataset


class _CursorDataset:
    """The shape every built-in adapter has: a shuffled list and an index."""

    def __init__(self, size: int = 100, batch_size: int = 4) -> None:
        self.records = list(range(size))
        self.batch_size = batch_size
        self.index = 0

    @property
    def n_prompts(self) -> int:
        return len(self.records)

    def next_batch(self, builder):
        del builder
        if self.index >= len(self.records):
            return []
        batch = self.records[self.index : self.index + self.batch_size]
        self.index += len(batch)
        return batch


class _SeekDataset(_CursorDataset):
    """An adapter whose cursor is not a plain attribute."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seeks: list[int] = []

    def seek(self, index: int) -> None:
        self.seeks.append(index)
        self.index = index


class _OpaqueDataset:
    def next_batch(self, builder):
        return []


# ── the dataset seek contract ────────────────────────────────────────────
def test_seek_prefers_the_explicit_hook_over_the_index_attribute():
    dataset = _SeekDataset()
    _seek_dataset(dataset, 40)
    assert dataset.seeks == [40]
    assert dataset.index == 40


def test_seek_falls_back_to_the_index_attribute():
    dataset = _CursorDataset()
    _seek_dataset(dataset, 40)
    assert dataset.index == 40


def test_a_dataset_with_neither_is_rejected_at_the_first_epoch():
    # Better here than at the first resume, which is hours in and after a
    # crash. ``_require_seekable`` runs when the first epoch's dataset is
    # built, not when a checkpoint is taken.
    with pytest.raises(TypeError, match="cannot resume"):
        _require_seekable(_OpaqueDataset())
    _require_seekable(_CursorDataset())
    _require_seekable(_SeekDataset())


def test_seeking_skips_the_prompts_already_handed_out():
    dataset = _CursorDataset(size=20, batch_size=4)
    seen = []
    for _ in range(3):
        seen.extend(dataset.next_batch(None))
    assert seen == list(range(12))

    # A crash here; the resumed run rebuilds the dataset from the same seed
    # (so the permutation is identical) and skips forward.
    resumed = _CursorDataset(size=20, batch_size=4)
    _seek_dataset(resumed, 12)
    rest = []
    while True:
        batch = resumed.next_batch(None)
        if not batch:
            break
        rest.extend(batch)
    assert rest == list(range(12, 20))
    assert not set(seen) & set(rest)


# ── snapshot / restore of the worker's counters ──────────────────────────
class _FakeRolloutWorker:
    """``RolloutWorker``'s checkpoint surface, without its TQ/engine deps.

    ``RolloutWorker.__init__`` builds a tokenizer and a TQ client, neither of
    which belongs in a counter test, so the two methods under test are bound
    onto a bare object holding the same attributes.
    """

    from meshy.worker.rollout import RolloutWorker as _Real

    _snapshot = _Real._snapshot
    _restore = _Real._restore
    store = _Real.store

    def __init__(self, store: StateStore, policy: GlobalCheckpointConfig, resume_step=0):
        self._store = store
        self._state_dir = store.root
        self.checkpoint = policy
        self.resume_step = int(resume_step)
        self.service_name = "rollout-0"
        self._epoch = 0
        self._dataset_index = 0
        self._filtered_at_epoch_start = 0
        self.groups_seen = 0
        self.groups_filtered = 0
        self.gates_seen = 0
        self.samples_started = 0
        self.weight_version = 0
        self._restore()


def test_snapshot_then_restore_round_trips_every_counter(tmp_path):
    store = StateStore(str(tmp_path))
    policy = GlobalCheckpointConfig(enabled=True, interval=10, keep_latest_k=2)

    before = _FakeRolloutWorker(store, policy)
    before._epoch = 1
    before._dataset_index = 960
    before._filtered_at_epoch_start = 3
    before.groups_seen = 620
    before.groups_filtered = 7
    before.gates_seen = 31  # gate 30 has just been consumed
    before.samples_started = 14880
    before.weight_version = 10
    before._snapshot(30)

    after = _FakeRolloutWorker(store, policy, resume_step=30)
    assert after._epoch == 1
    assert after._dataset_index == 960
    assert after._filtered_at_epoch_start == 3
    assert after.groups_seen == 620
    assert after.groups_filtered == 7
    assert after.samples_started == 14880
    assert after.weight_version == 10
    # One *below* the snapshot: the trainer re-publishes a genesis pulse at
    # step 30 to rebuild the gate stream TQ lost, and this worker has to be
    # expecting exactly that pulse.
    assert after.gates_seen == 30


def test_restore_refuses_a_step_with_no_shard(tmp_path):
    store = StateStore(str(tmp_path))
    policy = GlobalCheckpointConfig(enabled=True, interval=10, keep_latest_k=2)
    with pytest.raises(FileNotFoundError, match="shard is incomplete"):
        _FakeRolloutWorker(store, policy, resume_step=30)


def test_a_fresh_worker_restores_nothing(tmp_path):
    store = StateStore(str(tmp_path))
    worker = _FakeRolloutWorker(store, GlobalCheckpointConfig())
    assert (worker.gates_seen, worker._dataset_index, worker._epoch) == (0, 0, 0)


def test_snapshot_rotates_old_shards(tmp_path):
    store = StateStore(str(tmp_path))
    policy = GlobalCheckpointConfig(enabled=True, interval=10, keep_latest_k=2)
    worker = _FakeRolloutWorker(store, policy)
    for step in (10, 20, 30):
        worker.gates_seen = step + 1
        worker._snapshot(step)
    assert store.complete_steps("rollout-0") == [20, 30]
