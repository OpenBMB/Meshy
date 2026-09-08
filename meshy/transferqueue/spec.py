"""Derive a TransferQueue deployment spec from a recipe's ServiceGroups.

TransferQueue is mandatory infrastructure: every recipe's data plane (samples)
and control plane (gen gates) ride on it. A recipe *may* export a
``TRANSFER_QUEUE`` dict to override tuning knobs, but when it does not, the
launcher derives a sound spec from the typed per-role configs here.

Sizing rationale for ``pre_alloc_sample_num`` (allocated by the controller
**per partition**): the data partition must hold everything in flight at once
-- samples the trainer has not consumed yet (up to the pacing window's worth of
batches), plus rollouts still generating (``async_max_running_request``).
``max(4 * batch, 2 * max_running + batch, 1024)`` covers both regimes with
slack; undersizing
shows up as producers blocking on index allocation, never as corruption.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from meshy.transferqueue.client import resolve_endpoints_file

if TYPE_CHECKING:
    from meshy.service.base import ServiceGroup

_MIN_PRE_ALLOC = 1024


def derive_tq_spec(
    service_groups: "list[ServiceGroup]", runtime_dir: str
) -> dict[str, Any]:
    """Compute the TransferQueue launch spec for a recipe.

    Environment overrides:
    ``XRL_TQ_PRE_ALLOC``, ``XRL_TQ_STORAGE_UNITS``, ``XRL_TQ_STORAGE_SIZE``,
    ``XRL_TQ_ENDPOINTS``.
    """
    from meshy.config import RolloutServiceConfig, TrainingServiceConfig

    batch_size = 0
    max_running = 0
    for group in service_groups:
        config = group.config
        if isinstance(config, TrainingServiceConfig):
            batch_size = max(batch_size, int(config.batch_size))
        elif isinstance(config, RolloutServiceConfig):
            max_running = max(max_running, int(config.async_max_running_request))

    pre_alloc = max(4 * batch_size, 2 * max_running + batch_size, _MIN_PRE_ALLOC)

    return {
        "endpoints_file": resolve_endpoints_file(runtime_dir),
        "num_storage_units": int(os.environ.get("XRL_TQ_STORAGE_UNITS", "2")),
        "pre_alloc_sample_num": int(os.environ.get("XRL_TQ_PRE_ALLOC", str(pre_alloc))),
        "storage_unit_size": int(os.environ.get("XRL_TQ_STORAGE_SIZE", "100000")),
    }
