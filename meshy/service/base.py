"""Core Service / ServiceGroup abstractions for the card-level SPMD ignitor.

Every role in a run -- inference, training, rollout -- exists as a
**Service**. Two ideas carry the whole stack:

* :class:`ServiceGroup` -- a *declarative* description of a group of engine
  replicas. It pairs a stable ``id`` with a typed per-role config
  (:class:`~meshy.config.ServiceConfig` subclass, which also determines the
  role) plus the placement shape (``n_replicas`` / ``n_gpus_per_replica`` /
  ``colocate_with`` / ``wait_until``). It carries no runtime state; the
  :class:`~meshy.service.ignite.Ignitor` expands it into a flat list of per-card
  :class:`Service` objects.
* :class:`Service` -- the *runtime* entity that lives inside one ignitor
  process (one process per GPU card). ``ignite()`` spawns the actual Engine
  subprocess for this card; the Engine composes its role Worker, while the
  Service remains responsible for process lifecycle and readiness. ``wait_for_ready()``
  confirms readiness over HTTP and publishes the service marker into the
  shared runtime directory. :meth:`Service.from_info` is the per-role factory
  the ignitor calls, keeping the ignitor role-agnostic.

This module is intentionally free of heavy imports (no torch / sglang) so the
launcher can import a recipe and read its :class:`ServiceGroup` list cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from meshy.config import ServiceConfig

if TYPE_CHECKING:
    from meshy.service.runtime import RuntimeDir
    from meshy.service.topology import ServiceInfo, Topology


def die_with_parent(signal_number: int | None = None) -> None:
    """Ask the kernel to signal this process when its parent dies (Linux only).

    Spawned engine subprocesses (trainer / rollout) are non-daemonic; if the
    ignitor/launcher is killed abruptly (e.g. SIGKILL) they would be reparented
    to init and keep running -- holding GPU memory and TCP ports, which breaks
    the next run (``EADDRINUSE`` on the dist/HTTP ports, or a stale engine being
    mistaken for "ready"). SGLang already self-exits on parent death; our
    torchtitan engine does not, so it must opt in. ``PR_SET_PDEATHSIG`` makes the
    kernel deliver ``signal_number`` (default ``SIGKILL``) the moment the parent
    exits, however it dies. No-op on non-Linux platforms.
    """
    import platform
    import signal as _signal

    if platform.system() != "Linux":
        return
    sig = _signal.SIGKILL if signal_number is None else signal_number
    try:
        import ctypes

        _PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, sig)
    except Exception:
        pass


def raise_open_file_limit() -> int:
    """Raise this process' soft ``RLIMIT_NOFILE`` up to its hard limit (Linux).

    The Rollout legitimately holds *hundreds* of sockets open at once for its
    concurrent inference requests (plus, historically, the blocking trainer
    ``/samples`` POSTs). Under a low default soft limit (commonly 1024) this
    trips ``OSError: [Errno 24] Too many open files`` mid-batch. Bumping the
    soft limit to the hard limit is safe and reversible per-process. Returns
    the resulting soft limit (or ``-1`` on any non-POSIX platform / failure).
    """
    try:
        import resource
    except Exception:
        return -1
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and soft >= hard:
            return soft
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        return new_soft
    except Exception:
        return -1


def redirect_output(log_path: str) -> None:
    """Redirect this process' stdout and stderr to an append-only log file.

    This is intended for Ignitor and its spawned role processes. It deliberately
    operates on file descriptors rather than only replacing ``sys.stdout``.  That also
    captures loguru, native extensions, and subprocesses inheriting the role's
    standard streams.  The caller should invoke it at process startup.
    """
    import os
    import sys

    parent = os.path.dirname(os.path.abspath(log_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    log = open(log_path, "ab", buffering=0)
    try:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass
    finally:
        log.close()


@dataclass
class GPU:
    """A single GPU card discovered under card-level SPMD.

    ``global_rank`` is the torchrun global ``RANK`` and doubles as the card
    index used for deterministic assignment / port derivation. ``node_rank``
    is the torchrun ``GROUP_RANK`` (node index) and ``local_rank`` the
    ``LOCAL_RANK`` (device ordinal on that node).
    """

    host: str
    global_rank: int
    node_rank: int
    local_rank: int
    uuid: str = ""


@dataclass
class ServiceGroup:
    """Declarative node in the flat topology DAG for a group of engine replicas.

    Every group carries a stable :attr:`id` (unique across the recipe) so it
    can be referenced by other groups, and a typed per-role ``config`` whose
    class determines the role and runtime Service implementation. Whether the
    group consumes GPUs is declared by ``config.uses_gpu``.

    Colocation (two engines time-sharing the same physical cards) is expressed by
    :attr:`colocate_with`: a group with ``colocate_with=<other id>`` reuses that
    other group's exact card block instead of claiming fresh cards (the two only
    need to occupy the same number of cards; they may partition them
    asymmetrically, e.g. 8x TP=1 inference vs a single FSDP=8 trainer).

    Readiness dependencies are expressed by :attr:`wait_until`: a list of group
    ids that must all be ready before this group ignites. This is how e.g. a
    colocate trainer waits for the inference engines to free their GPU memory
    (``wait_until=[<inference id>]``), and how the rollout waits for both the
    trainer and inference to be up.
    """

    id: str
    config: ServiceConfig
    n_replicas: int = 1
    n_gpus_per_replica: int = 1
    colocate_with: str | None = None
    wait_until: list[str] = field(default_factory=list)

    @property
    def role(self) -> str:
        return type(self.config).role

    @property
    def uses_gpu(self) -> bool:
        return type(self.config).uses_gpu

    @property
    def n_gpus(self) -> int:
        """Physical cards this group *newly* claims (0 if it reuses another's)."""
        if not self.uses_gpu:
            return 0
        if self.colocate_with is not None:
            return 0
        return self.n_replicas * self.n_gpus_per_replica


class Service:
    """Per-card runtime entity managed by one ignitor process.

    Subclasses implement :meth:`from_info` (build the Service from its resolved
    :class:`~meshy.service.topology.ServiceInfo` -- this is where each role does
    its own topology wiring), :meth:`ignite` (spawn the engine subprocess) and
    :meth:`wait_for_ready` (poll the engine's HTTP API, then publish the
    readiness marker). The base class tracks spawned processes for lifecycle
    management.
    """

    def __init__(self, *, name: str, role: str) -> None:
        self.name = name
        self.role = role
        self.processes: list = []
        # Whether this card is the replica master (rank 0 within the replica).
        # Only the master polls readiness / publishes the readiness marker.
        self.is_master: bool = False
        self.colocation_config = None
        self.colocation_service_id: str | None = None
        self.is_colocation_leader = False
        self.colocation_endpoints_ref: str | None = None

    def configure_colocation(
        self,
        info: "ServiceInfo",
        runtime: "RuntimeDir",
    ) -> None:
        """Attach resolved ring metadata; role Services create the manager."""
        self.colocation_config = info.colocation
        self.colocation_service_id = info.group_id if info.colocation else None
        self.is_colocation_leader = bool(info.is_colocation_leader)
        if info.colocation is not None:
            from meshy.transferqueue.client import resolve_endpoints_file

            self.colocation_endpoints_ref = resolve_endpoints_file(runtime.root)

    def build_colocation_manager(self, *, on_acquire=None, on_release=None):
        """Build this Service's ring manager, or a no-op outside a ring.

        Callers remain responsible for starting the returned manager.  The
        factory lives on Service so role implementations do not duplicate TQ
        transport wiring and Engine stays unaware of control-plane details.
        """
        from meshy.service.colocation import ColocationManager, NoopColocationManager
        from meshy.transferqueue.colocation import TQRequestLedgerTransport

        if (
            self.colocation_config is None
            or not self.is_colocation_leader
            or self.colocation_service_id is None
            or self.colocation_endpoints_ref is None
        ):
            return NoopColocationManager()
        return ColocationManager(
            self.colocation_config,
            self.colocation_service_id,
            lambda: TQRequestLedgerTransport(
                self.colocation_endpoints_ref,
                self.colocation_config.group_id,
            ),
            on_acquire=on_acquire,
            on_release=on_release,
        )

    @classmethod
    def from_info(
        cls,
        info: "ServiceInfo",
        my_gpu: GPU | None,
        topology: "Topology",
        runtime: "RuntimeDir",
    ) -> "Service":  # pragma: no cover - interface
        raise NotImplementedError

    def ignite(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def wait_for_ready(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role}

    def join(self) -> None:
        for p in self.processes:
            p.join()

    def terminate(self) -> None:
        for p in self.processes:
            if p.is_alive():
                p.terminate()
