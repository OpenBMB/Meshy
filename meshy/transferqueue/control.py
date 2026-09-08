"""TransferQueue control plane: the gen-gate pulse that paces the Services.

The Services run as independent processes and coordinate only through TQ. The
control plane carries one signal: a per-step **gen gate** that the Training
Service raises to release the Rollout to generate a given step. The pulse
also carries the **weight version** the Rollout should generate against
(== the trainer step whose weights are currently live on the Inference
Service).

Why a gate is needed (colocate / lock-step): the Inference Service (SGLang) and
the Training Service share GPUs, so SGLang's memory is released while the
trainer holds the GPU. If the Rollout produced ``step_{i+1}`` early it would
hit SGLang ``/generate`` mid-training. So the Training Service only raises the
gate for ``i+1`` after ``step_i``'s weight sync restores SGLang. How far the
Rollout may run ahead of the newest gate is *its* policy (the pacing window,
see :class:`meshy.worker.rollout.RolloutWorker`); emission stays policy-free -- exactly
one pulse per weight version.

The pulses ride on TransferQueue itself (consistent with the data plane), as a
stream of single samples appended to **one rolling control partition**
``gen_gate``, each carrying ``(gate_step, weight_version)``. Per-step control
partitions were deliberately avoided: a 1-sample-1-column partition hits an
upstream hazard (``DataPartitionStatus.production_status`` is a shared
class-attribute tensor that only becomes per-partition once a multi-column
write triggers ``_expand_fields``),
and would also cost one full ``TQ_PRE_ALLOC_SAMPLE_NUM`` pre-allocation per
step. The rolling partition carries two columns (safe) and the consumer clears
each gate sample right after reading it, so control state stays O(1).

Consumption semantics do the ordering for us: the trainer is the only emitter
and appends gates in step order; the Rollout is the only consumer
(``task_name="gen_gate"``), reads them one at a time in production order, and
``read_gen_gate`` verifies the step stamp it got against the step it expected.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from loguru import logger
from tensordict import TensorDict

if TYPE_CHECKING:
    from transfer_queue import TransferQueueClient

#: rolling control partition all gen-gate pulses are appended to.
GEN_GATE_PARTITION = "gen_gate"
#: columns of one pulse. Two columns on purpose: single-column partitions alias
#: a shared production-status tensor upstream (see module docstring).
GEN_GATE_STEP_FIELD = "gate_step"
GEN_GATE_VERSION_FIELD = "weight_version"
GEN_GATE_FIELDS = [GEN_GATE_STEP_FIELD, GEN_GATE_VERSION_FIELD]
#: consumer identity; the Rollout is the gate stream's only reader.
GEN_GATE_TASK = "gen_gate"


def emit_gen_gate(
    client: "TransferQueueClient", step: int, weight_version: int = -1
):
    """Training Service: raise the gate that releases the Rollout for *step*.

    Appends one ``(gate_step, weight_version)`` sample to the rolling
    ``gen_gate`` partition. *weight_version* is the trainer step whose weights
    are live on the Inference Service. Returns the resulting ``BatchMeta``.
    """
    td = make_gen_gate(step=step, weight_version=weight_version)
    return client.put(data=td, partition_id=GEN_GATE_PARTITION)


def make_gen_gate(step: int, weight_version: int = -1) -> TensorDict:
    """Build one batched gen-gate payload without performing TQ I/O."""
    return TensorDict(
        {
            GEN_GATE_STEP_FIELD: torch.tensor([[int(step)]], dtype=torch.int64),
            GEN_GATE_VERSION_FIELD: torch.tensor(
                [[int(weight_version)]], dtype=torch.int64
            ),
        },
        batch_size=[1],
    )


def _scalar(value) -> int:
    if value.is_nested:
        value = value[0]
    return int(value.reshape(-1)[0].item())


def read_gen_gate(client: "TransferQueueClient", step: int) -> int | None:
    """Return the gate's weight version for *step*, or ``None`` if not yet raised.

    Uses ``mode="fetch"`` (not ``force_fetch``): fetch tolerates a not-yet-created
    partition (returns an empty BatchMeta under polling mode), whereas force_fetch
    raises "Partition not found" and dead-locks the caller. The fetch marks the
    gate sample consumed for the ``gen_gate`` task and the sample is cleared
    immediately after reading, so the stream stays O(1) regardless of run length.

    The caller reads gates strictly in step order; a mismatched ``gate_step``
    stamp means emitter and consumer have diverged and is logged loudly (the
    carried version is still returned -- it is the payload that matters).
    """
    meta = client.get_meta(
        data_fields=GEN_GATE_FIELDS,
        batch_size=1,
        partition_id=GEN_GATE_PARTITION,
        mode="fetch",
        task_name=GEN_GATE_TASK,
    )
    if meta.size < 1:
        return None
    td = client.get_data(meta)
    gate_step = _scalar(td[GEN_GATE_STEP_FIELD])
    version = _scalar(td[GEN_GATE_VERSION_FIELD])
    client.clear_samples(meta)
    if gate_step != int(step):
        logger.warning(
            "gen_gate stream out of order: expected step {}, got step {} "
            "(weight_version={})",
            step,
            gate_step,
            version,
        )
    return version


def wait_gen_gate(
    client: "TransferQueueClient",
    step: int,
    *,
    interval: float = 0.5,
    timeout: float | None = None,
) -> int:
    """Rollout: block until the gate for *step* is raised; return weight version.

    Args:
        client: connected TransferQueue client.
        step: step index to wait on.
        interval: polling interval in seconds.
        timeout: optional wall-clock timeout; raises ``TimeoutError`` if exceeded.

    Returns:
        The weight version carried by the gate (``-1`` when the producer did not
        set one).
    """
    start = time.monotonic()
    while True:
        version = read_gen_gate(client, step)
        if version is not None:
            return version
        if timeout is not None and (time.monotonic() - start) > timeout:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for gen gate step {step}."
            )
        time.sleep(interval)
