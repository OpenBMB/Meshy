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
