"""Parallelism-aware data distribution helpers.

These helpers wrap torchtitan's ``ParallelDims`` so the RL pipeline can scatter
samples correctly under any 5D combination of (PP, DP_replicate, DP_shard,
CP, TP) without hard-coding rank layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch.distributed as dist

if TYPE_CHECKING:
    from tensordict import TensorDict
    from torchtitan.distributed.parallel_dims import ParallelDims

    from .plan import MiniPlan, Plan


def dp_rank_and_size(parallel_dims: "ParallelDims") -> tuple[int, int]:
    """This rank's index within, and the size of, the ``batch`` (DP) mesh.

    The ``batch`` mesh is ``dp_replicate * dp_shard`` with CP/TP/PP factored
    out; it is ``None`` when both degrees are 1 (pure TP/CP run).
    """
    batch_mesh = parallel_dims.get_optional_mesh("batch")
    if batch_mesh is None:
        return 0, 1
    return batch_mesh.get_local_rank(), batch_mesh.size()


def split_batch_to_local(
    full_batch: "list[TensorDict] | None",
    parallel_dims: "ParallelDims",
    group=None,
    src: int = 0,
    planner: "Callable[[list[TensorDict], int], Plan] | None" = None,
) -> "tuple[list[TensorDict], tuple[MiniPlan, ...] | None]":
    """Broadcast ``full_batch`` from ``src`` and return this rank's local shard.

    Sharding rules under torchtitan's 5D parallelism:

    * **DP** (``dp_replicate * dp_shard``): split batch — each DP rank gets
      a disjoint subset.
    * **TP, PP**: replicate batch — every rank inside a TP/PP group must see
      the **same** samples. TP needs bit-wise identical inputs across the
      group so column/row-parallel matmuls produce consistent activations;
      PP only consumes raw input on stage 0 but later stages keep the same
      shard for any preprocessing they share.
    * **CP**: replicate batch at this layer; CP additionally splits the
      sequence dimension *inside* the trainer (``CpSharder``), which is
      independent of this sample-level scatter.

    With a ``planner`` (normally ``TitanTrainer.plan_batch``) the subset is
    chosen by the dynamic-batching plan — balanced by token count, with the
    per-rank mini/micro schedule attached — and every rank runs the planner
    on the same broadcast list, so no plan needs to be communicated. Without
    a planner the batch is cut into ``dp_size`` contiguous slices and the
    returned plan is ``None``.

    The function uses ``broadcast_object_list`` rather than
    ``scatter_object_list`` because the rank-to-DP-index mapping depends on
    torchtitan's internal mesh order ``(pp, dp_replicate, fsdp, tp)`` and is
    fragile to reproduce on rank 0. Broadcasting the full list is cheap for
    the small TensorDict objects used here and keeps the slicing
    logic local to each rank, where ``parallel_dims`` already knows the
    answer.

    Args:
        full_batch: On rank ``src``, the complete sample list. On other
            ranks, the value is ignored (pass ``None`` or anything).
        parallel_dims: The ``ParallelDims`` instance owned by ``ForgeEngine``
            (typically ``trainer.parallel_dims``).
        group: Process group used for the broadcast. ``None`` uses the
            default group; for the trainer/inference hand-off pattern in
            this repo, pass the gloo group so the call doesn't conflict
            with an in-flight NCCL stream.
        src: Source rank that owns the full batch. Defaults to 0.
        planner: Optional ``(full_batch, dp_size) -> Plan`` callable.

    Returns:
        ``(local_samples, local_plan)``. Same content across all ranks within
        the same TP/CP/PP group; disjoint across DP.
    """
    container: list = [full_batch] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(container, src=src, group=group)
    batch = container[0]
    assert batch is not None, (
        f"broadcast_object_list returned None on rank {dist.get_rank()}; "
        f"check that src={src} actually held the batch"
    )

    my_dp_rank, dp_size = dp_rank_and_size(parallel_dims)

    if planner is not None:
        plan = planner(batch, dp_size)
        return plan.local_samples(batch, my_dp_rank), plan.per_rank[my_dp_rank]

    assert len(batch) % dp_size == 0, (
        f"batch size {len(batch)} not divisible by DP size {dp_size}; "
        f"adjust rollout.batch_size * group_size to a multiple of "
        f"dp_replicate ({parallel_dims.dp_replicate}) * "
        f"dp_shard ({parallel_dims.dp_shard})"
    )
    n_per_dp = len(batch) // dp_size
    return batch[my_dp_rank * n_per_dp : (my_dp_rank + 1) * n_per_dp], None
