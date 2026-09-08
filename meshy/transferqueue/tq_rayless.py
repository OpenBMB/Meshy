# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tq_rayless — run TransferQueue as a standalone program without Ray.

Importing this package:

1. Installs a local-execution mock of ``ray`` into ``sys.modules`` (see
   :mod:`tq_rayless.ray_shim`). This MUST happen before ``transfer_queue`` is
   imported, because TransferQueue does ``import ray`` at module-load time.
2. Makes the vendored / referenced ``transfer_queue`` source importable.
3. Re-exports the full TransferQueue public API, so ``import tq_rayless as tq``
   is a drop-in replacement for ``import transfer_queue as tq`` — minus Ray.

The location of the TransferQueue source is resolved in this order:

* ``$TRANSFER_QUEUE_SRC`` environment variable (path containing a
  ``transfer_queue`` package), if set;
* the vendored git submodule at ``third_party/TransferQueue``;
* an already-installed ``transfer_queue`` package on ``sys.path``.
"""

from __future__ import annotations

import os
import sys

from . import ray_shim

# --------------------------------------------------------------------------- #
# Step 1: install the ray shim BEFORE any transfer_queue import.
# --------------------------------------------------------------------------- #
ray_shim.install()


# --------------------------------------------------------------------------- #
# Step 2: make the transfer_queue source importable.
# --------------------------------------------------------------------------- #
def _resolve_transfer_queue_path() -> str | None:
    """Return a directory to add to sys.path so ``transfer_queue`` imports, or None.

    None means: rely on an already-importable ``transfer_queue`` (installed pkg).
    """
    # (a) explicit override
    env_path = os.environ.get("TRANSFER_QUEUE_SRC")
    if env_path:
        candidate = os.path.abspath(os.path.expanduser(env_path))
        if os.path.isdir(os.path.join(candidate, "transfer_queue")):
            return candidate
        # Allow pointing directly at the inner package's parent.
        if os.path.basename(candidate) == "transfer_queue":
            return os.path.dirname(candidate)
        raise FileNotFoundError(
            f"TRANSFER_QUEUE_SRC={env_path!r} does not contain a 'transfer_queue' package."
        )

    # (b) vendored submodule
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    submodule = os.path.join(repo_root, "third_party", "TransferQueue")
    # Meshy vendors this package under ``meshy/vendor``; the sibling package
    # is placed directly alongside the shim there.
    bundled = os.path.dirname(here)
    if os.path.isdir(os.path.join(bundled, "transfer_queue")):
        return bundled
    if os.path.isdir(os.path.join(submodule, "transfer_queue")):
        return submodule

    # (c) fall back to an installed transfer_queue (no path injection needed)
    return None


_tq_path = _resolve_transfer_queue_path()
if _tq_path and _tq_path not in sys.path:
    sys.path.insert(0, _tq_path)


# --------------------------------------------------------------------------- #
# Step 3: import transfer_queue and re-export its public API.
# --------------------------------------------------------------------------- #
try:
    import transfer_queue as _tq
except ImportError as exc:  # pragma: no cover - actionable guidance
    raise ImportError(
        "Could not import 'transfer_queue'. Make sure the TransferQueue source is "
        "available via one of:\n"
        "  - set $TRANSFER_QUEUE_SRC to the repo path, or\n"
        "  - initialize the git submodule at third_party/TransferQueue "
        "(`git submodule update --init`), or\n"
        "  - install it without deps: `pip install --no-deps TransferQueue`.\n"
        f"Original error: {exc}"
    ) from exc

# High-Level KV Interface
init = _tq.init
close = _tq.close
get_metrics_endpoint = _tq.get_metrics_endpoint
kv_put = _tq.kv_put
kv_batch_put = _tq.kv_batch_put
kv_batch_get = _tq.kv_batch_get
kv_batch_get_by_meta = _tq.kv_batch_get_by_meta
kv_list = _tq.kv_list
kv_clear = _tq.kv_clear
async_kv_put = _tq.async_kv_put
async_kv_batch_put = _tq.async_kv_batch_put
async_kv_batch_get = _tq.async_kv_batch_get
async_kv_batch_get_by_meta = _tq.async_kv_batch_get_by_meta
async_kv_list = _tq.async_kv_list
async_kv_clear = _tq.async_kv_clear
KVBatchMeta = _tq.KVBatchMeta

# High-Level StreamingDataLoader Interface
StreamingDataset = _tq.StreamingDataset
StreamingDataLoader = _tq.StreamingDataLoader

# Low-Level Native Interface
get_client = _tq.get_client
BatchMeta = _tq.BatchMeta
TransferQueueClient = _tq.TransferQueueClient

# Samplers
BaseSampler = _tq.BaseSampler
GRPOGroupNSampler = _tq.GRPOGroupNSampler
SequentialSampler = _tq.SequentialSampler
RankAwareSampler = _tq.RankAwareSampler
SeqlenBalancedSampler = _tq.SeqlenBalancedSampler

# tq_rayless-specific launcher helpers
from . import launcher  # noqa: E402
from .launcher import (  # noqa: E402
    build_config,
    connect,
    connect_endpoints,
    endpoints_to_config,
    start_local,
)

__all__ = [
    # High-Level KV Interface
    "init",
    "close",
    "get_metrics_endpoint",
    "kv_put",
    "kv_batch_put",
    "kv_batch_get",
    "kv_batch_get_by_meta",
    "kv_list",
    "kv_clear",
    "async_kv_put",
    "async_kv_batch_put",
    "async_kv_batch_get",
    "async_kv_batch_get_by_meta",
    "async_kv_list",
    "async_kv_clear",
    "KVBatchMeta",
    # StreamingDataLoader
    "StreamingDataset",
    "StreamingDataLoader",
    # Low-Level
    "get_client",
    "BatchMeta",
    "TransferQueueClient",
    # Samplers
    "BaseSampler",
    "GRPOGroupNSampler",
    "SequentialSampler",
    "RankAwareSampler",
    "SeqlenBalancedSampler",
    # tq_rayless launcher
    "launcher",
    "start_local",
    "build_config",
    "connect",
    "connect_endpoints",
    "endpoints_to_config",
    "ray_shim",
]

__version__ = "0.1.0"
