"""The unified GRPO data plane over a real (in-process) TransferQueue.

After the cutover every driver's samples flow agentloop -> TQ -> trainer. The
GRPO contract this pins down:

* the agentloop's five-column put (incl. the ``weight_version`` staleness tag)
  round-trips losslessly, with ``weight_version`` staying int64;
* the trainer's AND-filter fetch returns complete samples only, and honours
  the ``batch_size`` granularity (an under-filled window reads as empty);
* ``clear_samples`` recycles indexes so a run of many rounds stays bounded.

Run with the torchtitan env (torch + tensordict + tq_rayless):

    TRANSFER_QUEUE_SRC=/workspace/TransferQueue \\
    python -m pytest tests/test_tq_grpo_pipeline.py -q
"""

from __future__ import annotations

import os

BATCH = 8
os.environ.setdefault("TQ_PRE_ALLOC_SAMPLE_NUM", str(BATCH))

import pytest  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from meshy.config import GRPO_TRAINER_FIELDS  # noqa: E402
from meshy.transferqueue import adapter  # noqa: E402
from meshy.transferqueue.client import import_tq  # noqa: E402


@pytest.fixture(scope="module")
def client():
    tq = import_tq()
    tq.start_local(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "SimpleStorage": {"num_data_storage_units": 2, "total_storage_size": 512}
            },
        }
    )
    c = tq.get_client()
    yield c
    tq.close()


def make_grpo_samples(batch: int = BATCH, seed: int = 0, version: int = 3) -> list[TensorDict]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(batch):
        length = 3 + i
        out.append(
            TensorDict(
                {
                    "tokens": torch.randint(0, 1000, (length,), generator=g, dtype=torch.long),
                    "logprobs": -torch.rand(length, generator=g, dtype=torch.float32),
                    "mask_assistant": torch.ones(length, dtype=torch.float32),
                    "advantage": torch.tensor(float(i) - batch / 2, dtype=torch.float32),
                    "weight_version": torch.tensor(version, dtype=torch.int64),
                    "reward": torch.tensor(float(i % 2), dtype=torch.float32),
                    "truncated": torch.tensor(i % 3 == 0, dtype=torch.int64),
                    "repetition": torch.tensor(0, dtype=torch.int64),
                    "mixed_version": torch.tensor(i % 2, dtype=torch.int64),
                },
                batch_size=[],
            )
        )
    return out


def test_weight_version_roundtrips_as_int64():
    samples = make_grpo_samples(version=41)
    td = adapter.samples_to_td(samples, GRPO_TRAINER_FIELDS)
    assert td["weight_version"].dtype == torch.int64
    back = adapter.td_to_samples(td, GRPO_TRAINER_FIELDS)
    for s in back:
        assert s["weight_version"].dtype == torch.int64
        assert int(s["weight_version"]) == 41


def test_trainer_fetch_honours_batch_granularity(client):
    """A half-filled window must read as empty, not as a short batch."""
    partition = "grpo@gran"
    half = make_grpo_samples(batch=BATCH // 2, seed=1)
    client.put(data=adapter.samples_to_td(half, GRPO_TRAINER_FIELDS), partition_id=partition)

    early = client.get_meta(
        data_fields=GRPO_TRAINER_FIELDS,
        batch_size=BATCH,
        partition_id=partition,
        mode="fetch",
        task_name="trainer",
    )
    assert early.size == 0, "trainer got a short batch"

    rest = make_grpo_samples(batch=BATCH - BATCH // 2, seed=2)
    client.put(data=adapter.samples_to_td(rest, GRPO_TRAINER_FIELDS), partition_id=partition)
    full = client.get_meta(
        data_fields=GRPO_TRAINER_FIELDS,
        batch_size=BATCH,
        partition_id=partition,
        mode="fetch",
        task_name="trainer",
    )
    assert full.size == BATCH
    got = adapter.td_to_samples(client.get_data(full), GRPO_TRAINER_FIELDS)
    assert len(got) == BATCH
    for s in got:
        assert set(GRPO_TRAINER_FIELDS) <= set(s.keys())
    client.clear_partition(partition)


def test_clear_samples_recycles_indexes_across_rounds(client):
    """Consume-and-clear must keep working far past the pre-allocation."""
    partition = "grpo@rounds"
    pre_alloc = int(os.environ["TQ_PRE_ALLOC_SAMPLE_NUM"])
    rounds = 3 * max(1, pre_alloc // BATCH)
    for r in range(rounds):
        want = make_grpo_samples(seed=100 + r, version=r)
        client.put(
            data=adapter.samples_to_td(want, GRPO_TRAINER_FIELDS), partition_id=partition
        )
        meta = client.get_meta(
            data_fields=GRPO_TRAINER_FIELDS,
            batch_size=BATCH,
            partition_id=partition,
            mode="fetch",
            task_name="trainer",
        )
        assert meta.size == BATCH, f"round {r}: trainer starved"
        got = adapter.td_to_samples(client.get_data(meta), GRPO_TRAINER_FIELDS)
        for i, (g, w) in enumerate(zip(got, want)):
            assert torch.equal(g["tokens"], w["tokens"]), (r, i)
            assert int(g["weight_version"]) == r, (r, i)
        client.clear_samples(meta)
    client.clear_partition(partition)
