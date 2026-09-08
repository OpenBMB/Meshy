"""OPD data plane over an in-process TransferQueue.

Pins the sequencing contract that keeps the roles decoupled:

* the rollout writes plain GRPO rows and never learns about the Teacher;
* the Student's AND-filter fetch stays empty until the Teacher has appended
  its two columns to every row of the window;
* a GRPO-style consumer of the same rows is unaffected by the extra columns.

    TRANSFER_QUEUE_SRC=/workspace/TransferQueue \\
    python -m pytest tests/test_opd_pipeline.py -q
"""

from __future__ import annotations

import os

BATCH = 8
os.environ.setdefault("TQ_PRE_ALLOC_SAMPLE_NUM", str(BATCH))

import pytest  # noqa: E402
import torch  # noqa: E402

from meshy.config import GRPO_TRAINER_FIELDS, OPD_TEACHER_FIELDS, OPD_TRAINER_FIELDS  # noqa: E402
from meshy.worker.opd import OPDTeacherWorker  # noqa: E402
from meshy.transferqueue import adapter  # noqa: E402
from meshy.transferqueue.client import import_tq  # noqa: E402
from tests.test_tq_grpo_pipeline import make_grpo_samples  # noqa: E402

K = 4


@pytest.fixture(scope="module")
def client():
    tq = import_tq()
    tq.start_local({
        "controller": {"polling_mode": True},
        "backend": {"SimpleStorage": {"num_data_storage_units": 2, "total_storage_size": 512}},
    })
    c = tq.get_client()
    yield c
    tq.close()


class _FakeTeacher:
    """Scores rows deterministically from their tokens (no model)."""

    name = "teacher"

    def __init__(self):
        self.calls = []

    def score(self, samples):
        self.calls.append([td["tokens"].clone() for td in samples])
        out = []
        for td in samples:
            L = int(td["tokens"].numel())
            ids = torch.stack([td["tokens"] + i for i in range(K)], dim=-1)
            out.append({"teacher_topk_ids": ids, "teacher_topk_logprobs": -ids.float() / 1000})
        return out


class _Ring:
    def __init__(self):
        self.events = []

    def request_gpu(self, request_id=None, **kw):
        self.events.append(("request", request_id))
        return request_id

    def wait_for_grant(self, request):
        self.events.append(("grant", request))

    def release(self, *, transition="", payload_ref=None):
        self.events.append(("release", transition, payload_ref))


def _fetch(client, fields, task, partition, batch=BATCH):
    return client.get_meta(data_fields=list(fields), batch_size=batch, partition_id=partition,
                           mode="fetch", task_name=task)


def test_student_fetch_waits_for_teacher_columns(client):
    partition = "opd@seq"
    rows = make_grpo_samples(batch=BATCH, seed=7, version=2)
    client.put(data=adapter.samples_to_td(rows, GRPO_TRAINER_FIELDS), partition_id=partition)

    # Student: nothing yet, the Teacher columns do not exist.
    assert _fetch(client, OPD_TRAINER_FIELDS, "titan", partition).size == 0

    teacher = _FakeTeacher()
    ring = _Ring()
    worker = OPDTeacherWorker(
        engine=teacher, endpoints_ref="in-process", partition_id=partition,
        score_batch_size=BATCH // 2, colocation=ring,
        student_weights=lambda v: f"/ckpt/actor_train/v{v}", client_factory=lambda ref: client,
    )
    assert worker.run_tq_once(client)  # first window
    assert _fetch(client, OPD_TRAINER_FIELDS, "titan", partition).size == 0, "half a batch leaked"
    assert worker.run_tq_once(client)  # second window
    assert not worker.run_tq_once(client)  # nothing left for the Teacher
    assert len(teacher.calls) == 2 and all(len(c) == BATCH // 2 for c in teacher.calls)
    assert ring.events[2] == ("release", "teacher-score-complete", "/ckpt/actor_train/v2")

    # Student: the union is complete now, columns come back as [L, K].
    meta = _fetch(client, OPD_TRAINER_FIELDS, "titan", partition)
    assert meta.size == BATCH
    got = adapter.td_to_samples(client.get_data(meta), OPD_TRAINER_FIELDS)
    for td in got:
        L = int(td["tokens"].numel())
        assert td["teacher_topk_ids"].shape == (L, K)
        assert td["teacher_topk_logprobs"].shape == (L, K)
        assert torch.equal(td["teacher_topk_ids"][:, 0], td["tokens"])
        assert td["weight_version"].dtype == torch.int64
    client.clear_samples(meta)
    client.clear_partition(partition)


def test_grpo_consumer_ignores_teacher_columns(client):
    partition = "opd@grpo"
    rows = make_grpo_samples(batch=BATCH, seed=9)
    client.put(data=adapter.samples_to_td(rows, GRPO_TRAINER_FIELDS), partition_id=partition)
    worker = OPDTeacherWorker(engine=_FakeTeacher(), endpoints_ref="in-process", partition_id=partition,
                              score_batch_size=BATCH, client_factory=lambda ref: client)
    assert worker.run_tq_once(client)
    meta = _fetch(client, GRPO_TRAINER_FIELDS, "grpo", partition)
    assert meta.size == BATCH
    got = adapter.td_to_samples(client.get_data(meta), GRPO_TRAINER_FIELDS)
    assert all(set(td.keys()) == set(GRPO_TRAINER_FIELDS) for td in got)
    assert not (set(OPD_TEACHER_FIELDS) & set(got[0].keys()))
    client.clear_samples(meta)
    client.clear_partition(partition)
