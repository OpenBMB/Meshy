"""Run-level bootstrap KV plane (a ``torch.distributed.TCPStore``).

Startup-time cross-process synchronization used to ride on two kinds of files
in the shared runtime directory: readiness markers (``ready/{name}.ready``) and
the TransferQueue endpoints JSON. Both are one-shot key/value publications
("service X is ready", "the TQ endpoints are Y"), which is exactly what a
TCPStore provides -- ``set`` / ``wait`` / ``check`` -- over plain TCP, so
multi-node runs no longer need a shared filesystem for synchronization (only
on-disk artifacts such as weight checkpoints still do).

Exactly one process per run hosts the store; everyone else connects as a
client. The resolution rule, shared by every process:

1. ``XRL_BOOTSTRAP_ADDR=host:port`` is set -> connect as a client. The hosting
   process exports this variable *before* spawning children, so children can
   never accidentally try to host.
2. Under torchrun (``MASTER_ADDR``/``MASTER_PORT`` present) -> the address is
   ``MASTER_ADDR:MASTER_PORT+1`` (port overridable via ``XRL_BOOTSTRAP_PORT``)
   and global rank 0 hosts. This covers the direct ``torchrun -m recipe.x``
   path, where no launcher exists.
3. Otherwise (module smoke tests, pytest) -> host a store on a free localhost
   port and export ``XRL_BOOTSTRAP_ADDR`` so subprocesses spawned later become
   clients of it.

``scripts/launch.py`` short-circuits the rule by hosting explicitly on node 0
(:func:`host_store`) and exporting ``XRL_BOOTSTRAP_ADDR`` for the TQ processes
and all torchrun children (on every node, so multi-node workers reach node 0's
store instead of a shared file).

The store's lifetime is its host process' lifetime. That is the semantics we
want: if the launcher / rank-0 ignitor dies the run is dead anyway, and a fresh
run gets a fresh store -- the "stale endpoints file from a previous run" hazard
cannot exist here. Keys are still namespaced by runtime root (see callers)
because one *process* may serve several runs back-to-back (pytest).
"""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import timedelta
from typing import Any, Iterable

# How long a client waits for the host to appear before failing loudly. The
# host is the first thing a run brings up, so a long wait here almost always
# means "the launcher is not running", not "the launcher is slow".
CLIENT_CONNECT_TIMEOUT_S = float(os.environ.get("XRL_BOOTSTRAP_TIMEOUT", "60"))

_store = None  # process-wide singleton (TCPStore client or master)


def parse_addr(addr: str) -> tuple[str, int]:
    host, _, port = addr.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"XRL_BOOTSTRAP_ADDR must be 'host:port', got {addr!r}")
    return host, int(port)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _tcp_store(host: str, port: int, *, is_master: bool):
    from torch.distributed import TCPStore

    if is_master:
        return TCPStore(host, port, is_master=True, wait_for_workers=False)
    try:
        return TCPStore(
            host,
            port,
            is_master=False,
            timeout=timedelta(seconds=CLIENT_CONNECT_TIMEOUT_S),
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not reach the bootstrap store at {host}:{port} within "
            f"{CLIENT_CONNECT_TIMEOUT_S:.0f}s -- is the hosting process "
            "(scripts/launch.py or the rank-0 ignitor) alive? "
            "(XRL_BOOTSTRAP_ADDR / XRL_BOOTSTRAP_PORT control the address)"
        ) from exc


def host_store(host: str, port: int):
    """Explicitly host the run's store here and export its address.

    Used by ``scripts/launch.py`` on node 0. Must be called before any child
    process that will use the store is spawned.
    """
    global _store
    if _store is not None:
        return _store
    try:
        _store = _tcp_store(host, port, is_master=True)
    except Exception as exc:
        raise RuntimeError(
            f"could not host the bootstrap store on {host}:{port} "
            "(port in use from a previous run? override with XRL_BOOTSTRAP_PORT)"
        ) from exc
    os.environ["XRL_BOOTSTRAP_ADDR"] = f"{host}:{port}"
    return _store


def get_store():
    """Resolve the run's bootstrap store (see module docstring for the rule)."""
    global _store
    if _store is not None:
        return _store

    addr = os.environ.get("XRL_BOOTSTRAP_ADDR")
    if addr:
        _store = _tcp_store(*parse_addr(addr), is_master=False)
        return _store

    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    if master_addr and master_port:
        port = int(os.environ.get("XRL_BOOTSTRAP_PORT", int(master_port) + 1))
        is_master = int(os.environ.get("RANK", "0")) == 0
        _store = _tcp_store(master_addr, port, is_master=is_master)
        # Children of this process (engine subprocesses, the rollout driver)
        # must connect as clients even after torchrun env scrubbing.
        os.environ["XRL_BOOTSTRAP_ADDR"] = f"{master_addr}:{port}"
        return _store

    # Standalone owner: module smoke tests / pytest. Host locally and export
    # the address so subprocesses spawned from here inherit it.
    return host_store("127.0.0.1", _free_port())


# ── KV helpers ───────────────────────────────────────────────────────────────


def mark(key: str) -> None:
    get_store().set(key, "1")


def set_json(key: str, obj: Any) -> None:
    get_store().set(key, json.dumps(obj))


def get_json(key: str) -> Any:
    return json.loads(get_store().get(key).decode("utf-8"))


def check(keys: Iterable[str]) -> bool:
    """True iff every key exists (non-blocking)."""
    return get_store().check(list(keys))


def wait_keys(
    keys: Iterable[str],
    timeout: float = 1800.0,
    interval: float = 1.0,
    liveness: "callable | None" = None,
) -> None:
    """Block until every key exists.

    Polls non-blocking ``check`` at ``interval`` rather than one big
    ``store.wait``: a chunked ``wait`` spams c10d timeout warnings on every
    slice, and polling lets a caller-provided ``liveness`` probe (e.g. "is the
    process that should publish this key still alive?") fail fast instead of
    burning the full timeout.
    """
    pending = list(dict.fromkeys(keys))
    if not pending:
        return
    store = get_store()
    deadline = time.monotonic() + timeout
    while True:
        pending = [k for k in pending if not store.check([k])]
        if not pending:
            return
        if liveness is not None:
            liveness()
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"bootstrap keys not published within {timeout}s: {pending} "
                f"(store at {os.environ.get('XRL_BOOTSTRAP_ADDR', '<in-process>')})"
            )
        time.sleep(interval)
