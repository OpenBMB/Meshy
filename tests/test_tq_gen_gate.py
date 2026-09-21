"""Control-plane checks: the gen-gate stream over a real (in-process) TQ.

The gen gate is the one signal that paces the whole system, so its semantics
are pinned down against a real controller + storage rather than a fake:

* an unraised gate reads as ``None`` (never raises, never blocks) -- this is
  what lets the AgentLoop poll it opportunistically;
* a raised gate is read **exactly once** (per-task consumption tracking) and
  carries its weight version intact;
* gates come back in emission order across many steps;
* the stream stays O(1): each gate sample is cleared right after its read, so
  a long run does not accumulate control samples.

Run with the torchtitan env (torch + tensordict + tq_rayless):

    TRANSFER_QUEUE_SRC=/workspace/TransferQueue \\
    python -m pytest tests/test_tq_gen_gate.py -q
"""

from __future__ import annotations

import os

# Must be set before the controller creates its first partition (read at
# partition-creation time). The gate stream holds at most a handful of samples.
os.environ.setdefault("TQ_PRE_ALLOC_SAMPLE_NUM", "8")

import pytest  # noqa: E402

from meshy.transferqueue import control  # noqa: E402
from meshy.transferqueue.client import import_tq  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """One in-process TQ system for the whole module (polling mode mandatory:
    without it an under-filled get_meta raises TimeoutError instead of
    reporting "not yet")."""
    tq = import_tq()
    tq.start_local(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "SimpleStorage": {"num_data_storage_units": 1, "total_storage_size": 64}
            },
        }
    )
    c = tq.get_client()
    yield c
    tq.close()


def test_unraised_gate_reads_as_none(client):
    assert control.read_gen_gate(client, 0) is None


def test_gate_roundtrip_and_single_consumption(client):
    control.emit_gen_gate(client, step=0, weight_version=0)
    assert control.read_gen_gate(client, 0) == 0
    # The single pulse was consumed (and cleared); the next poll sees nothing.
    assert control.read_gen_gate(client, 1) is None


def test_gates_arrive_in_emission_order(client):
    for step in (1, 2, 3):
        control.emit_gen_gate(client, step=step, weight_version=step)
    assert control.read_gen_gate(client, 1) == 1
    assert control.read_gen_gate(client, 2) == 2
    assert control.read_gen_gate(client, 3) == 3
    assert control.read_gen_gate(client, 4) is None


def test_gate_stream_stays_bounded(client):
    """Every read clears its sample, so index recycling must keep working far
    past TQ_PRE_ALLOC_SAMPLE_NUM total gates."""
    pre_alloc = int(os.environ["TQ_PRE_ALLOC_SAMPLE_NUM"])
    base = 10
    for i in range(3 * pre_alloc):
        step = base + i
        control.emit_gen_gate(client, step=step, weight_version=step)
        assert control.read_gen_gate(client, step) == step


def test_wait_gen_gate_times_out_cleanly(client):
    with pytest.raises(TimeoutError):
        control.wait_gen_gate(client, step=999, interval=0.05, timeout=0.3)


# ── resume ───────────────────────────────────────────────────────────────
class _StubEngine:
    name = "actor_train-0"

    def __init__(self, step_index: int) -> None:
        self.step_index = step_index
        self.weight_version = step_index


def _worker(*, resume_step: int, cold_start: int, step_index: int):
    from meshy.worker.titan import TitanWorker

    return TitanWorker(
        engine=_StubEngine(step_index),
        endpoints_ref="unused",
        partition_id="data.train",
        tq_fields=["tokens"],
        batch_size=8,
        gate_step_offset=cold_start,
        resume_step=resume_step,
    )


def _gate_step(outputs) -> int:
    return int(outputs["gate"]["gate_step"].item())


def test_genesis_gate_is_zero_on_a_fresh_run():
    # Gate 0 establishes the version inference may generate against, and it
    # precedes the critic's cold-start gates 1..N, so it is never offset.
    worker = _worker(resume_step=0, cold_start=20, step_index=0)
    assert _gate_step(worker.startup_tq_outputs()) == 0


def test_genesis_gate_re_establishes_the_stream_at_the_resume_step():
    # TQ is RAM-only, so a resumed run's gen_gate partition starts empty and
    # this same pulse has to rebuild the stream *where the run stopped*.
    # Numbering it 0 again would rewind every consumer's pacing budget.
    worker = _worker(resume_step=35, cold_start=20, step_index=15)
    assert _gate_step(worker.startup_tq_outputs()) == 35


def test_the_first_gate_after_a_resume_follows_the_genesis_pulse():
    # The trainer's step counter lags the global clock by the cold start, so
    # step 16 must number its gate 36 -- one past the resume pulse, with no
    # hole and no repeat for the rollout's ordering check to warn about.
    worker = _worker(resume_step=35, cold_start=20, step_index=15)
    assert _gate_step(worker.startup_tq_outputs()) == 35
    worker.engine.step_index = 16
    worker.engine.weight_version = 16
    assert _gate_step(worker._gate_output(step=16)) == 36
