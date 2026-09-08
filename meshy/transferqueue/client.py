"""Import shim and connection helpers for the ray-free TransferQueue.

``tq_rayless`` is a companion package that runs upstream TransferQueue without
Ray: it installs a local-execution mock of ``ray`` *before* ``transfer_queue`` is
imported, so the ``@ray.remote`` controller / storage classes become plain
in-process objects with their own ZMQ servers. Ray is used upstream only for
bootstrap and endpoint discovery -- never on the data path -- so nothing is lost.

Two things must be true before ``import tq_rayless`` succeeds, and both are
handled by :func:`resolve_source`:

1. ``ray`` must NOT be installed (the shim provides it; a real Ray install would
   shadow the shim and drag in the dependency we are avoiding);
2. the upstream ``transfer_queue`` source must be locatable. ``tq_rayless``
   resolves it as ``$TRANSFER_QUEUE_SRC`` -> vendored ``third_party/TransferQueue``
   submodule -> installed package. The vendored submodule is typically not
   checked out, so we default ``TRANSFER_QUEUE_SRC`` from ``XRL_TQ_SRC``.

Meshy always runs TQ in *multi-process* mode: the controller and storage units
are standalone processes (see :mod:`meshy.transferqueue.launch`) and every worker
-- rollout and trainer -- calls :func:`connect` against the shared
endpoints ref (a ``store://`` bootstrap-store key by default, or a JSON file for
an externally managed TQ). Single-process ``start_local`` is deliberately not
exposed: it only builds in-process actors, which the other worker processes
cannot reach.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep torch / tensordict / zmq out of module import time
    from omegaconf import DictConfig
    from transfer_queue import TransferQueueClient

# Default location of the upstream TransferQueue checkout. Overridable via
# ``XRL_TQ_SRC`` (Meshy-facing) or ``TRANSFER_QUEUE_SRC`` (tq_rayless-facing).
def import_tq():
    """Import and return the ``tq_rayless`` module (installing the ray shim)."""
    from meshy.transferqueue import tq_rayless
    return tq_rayless


def build_config(
    endpoints_ref: str, base_conf: "dict | DictConfig | None" = None
) -> "DictConfig":
    """Build a TQ config (populated ZMQ endpoints) from an endpoints ref."""
    if is_store_ref(endpoints_ref):
        return import_tq().endpoints_to_config(
            _fetch_store_endpoints(endpoints_ref), base_conf, source=endpoints_ref
        )
    return import_tq().build_config(endpoints_ref, base_conf)


def connect(
    endpoints_ref: str, base_conf: "dict | DictConfig | None" = None
) -> "TransferQueueClient":
    """Connect a client to the running TQ described by ``endpoints_ref``.

    The ref is either a ``store://<key>`` bootstrap-store ref (default) or a
    path to a shared JSON endpoints file (externally managed TQ).

    Does not call ``init()``: the client talks straight to the controller and
    storage units over ZMQ. Each process should hold exactly one client; the
    client owns a background event loop and a set of ZMQ sockets.
    """
    if is_store_ref(endpoints_ref):
        return import_tq().connect_endpoints(
            _fetch_store_endpoints(endpoints_ref), base_conf, source=endpoints_ref
        )
    return import_tq().connect(endpoints_ref, base_conf)


# ── bootstrap-store refs ────────────────────────────────────────────────────

STORE_SCHEME = "store://"


def is_store_ref(ref: str) -> bool:
    return ref.startswith(STORE_SCHEME)


def store_key(ref: str) -> str:
    """The bootstrap-store key a ``store://`` ref names."""
    return ref[len(STORE_SCHEME):]


def store_ref(runtime_root: str) -> str:
    """Canonical bootstrap-store ref for a run's TQ endpoints."""
    return f"{STORE_SCHEME}tq|{os.path.abspath(runtime_root)}"


def _fetch_store_endpoints(ref: str, timeout: float = 300.0) -> dict:
    """Block until the cluster publishes the consolidated roster, then read it.

    The cluster writes the ref key only after the controller and *every*
    storage unit are up (see ``TransferQueueCluster._start_store``), so unlike
    the incrementally-written endpoints file a partial storage roster -- which
    silently mis-routes because placement is ``global_idx % num_units`` -- is
    unobservable here.
    """
    from meshy.service import bootstrap

    key = store_key(ref)
    try:
        bootstrap.wait_keys([key], timeout=timeout, interval=0.5)
    except TimeoutError:
        raise TimeoutError(
            f"TransferQueue not ready within {timeout}s: no roster published at {ref}"
        ) from None
    return bootstrap.get_json(key)


def wait_for_endpoints(
    endpoints_ref: str,
    num_storage_units: int,
    timeout: float = 300.0,
    interval: float = 0.5,
) -> None:
    """Block until the controller and all storage units have registered.

    For a ``store://`` ref this waits on the consolidated roster key (which is
    complete by construction, so ``num_storage_units`` is not needed). For a
    file, the endpoints JSON is written incrementally (controller first, then
    one entry per storage rank) and read-modify-write'd by each publisher, so a
    client that connects too early would see a partial roster and build a
    storage manager that misses units. Since routing is ``global_idx %
    num_units``, a short roster is not a transient inconvenience -- it silently
    changes data placement.
    """
    if is_store_ref(endpoints_ref):
        _fetch_store_endpoints(endpoints_ref, timeout=timeout)
        return

    import json
    import time

    deadline = time.monotonic() + timeout
    while True:
        try:
            with open(endpoints_ref) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        storage = data.get("storage") or {}
        if data.get("controller") and len(storage) >= num_storage_units:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"TransferQueue not ready within {timeout}s: {endpoints_ref} has "
                f"controller={bool(data.get('controller'))} "
                f"storage={len(storage)}/{num_storage_units}"
            )
        time.sleep(interval)


def endpoints_path(runtime_root: str) -> str:
    """Location of the endpoints *file* inside a runtime directory (file mode)."""
    return os.path.join(runtime_root, "tq_endpoints.json")


def resolve_endpoints_file(runtime_root: str) -> str:
    """The endpoints ref a Service should connect to for this run.

    ``XRL_TQ_ENDPOINTS`` overrides (e.g. an externally managed TQ publishing to
    a shared JSON file, or another ``store://`` ref); otherwise the canonical
    per-run bootstrap-store ref. The launcher's spec derivation uses the same
    rule, so every process agrees on one ref.
    """
    return os.environ.get("XRL_TQ_ENDPOINTS") or store_ref(runtime_root)
