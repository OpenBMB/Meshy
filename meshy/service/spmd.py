"""Per-card Service template for SPMD engine roles.

:class:`SpmdService` is the Ignition-layer counterpart of
:class:`~meshy.engine.spmd.SpmdEngine`: replica bookkeeping (which card is
master, where the replica's endpoint lives), readiness signaling, and the
shared child-process entry that wires the three runtime layers together::

    build_engine() -> engine.initialize()          # Engine built by the Service
    build_colocation_manager()                     # ring manager (Noop outside)
    build_worker(engine, colocation)               # Engine handed to the Worker
    engine command loop + worker TQ thread

Role Services only implement the two ``build_*`` factories; everything else --
spawn, colocation wiring and thread startup are owned here so the
Service layer stays pure lifecycle.
"""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

from loguru import logger

from meshy.service.base import GPU, Service, die_with_parent, redirect_output
from meshy.service.runtime import RuntimeDir

if TYPE_CHECKING:
    from meshy.engine.spmd import SpmdEngine
    from meshy.service.topology import ServiceInfo
    from meshy.worker.tq import TQWorker


def require_ring(info: "ServiceInfo") -> None:
    """Colocate without a ring cannot arbitrate the GPU -- fail at wiring time.

    A colocated Service that is not a ring member would simply compute whenever
    a batch arrives, land on a card the inference engine still occupies, and
    OOM (or silently corrupt a run) hours in. The legacy HTTP arbitration path
    that used to cover this arrangement was removed; the declaration is now
    part of the recipe contract.
    """
    if info.is_colocate and info.colocation is None:
        raise ValueError(
            f"service group {info.group_id!r} is colocated (colocate_with / "
            "shared card block) but is not a member of any COLOCATIONS ring. "
            "GPU hand-off on shared cards is scheduled by the colocation ring; "
            "declare a ColocationRing covering this group and pass it to "
            "Ignitor(SERVICE_GROUPS, COLOCATIONS)."
        )


class SpmdService(Service):
    """Per-card Service wrapper shared by the SPMD engine roles.

    Owns the replica bookkeeping and the common ``wait_for_ready`` -- wait for
    the child runtime to signal initialization complete, then publish the
    readiness marker -- plus the generic
    :meth:`_run_runtime` child-process template.
    """

    def __init__(
        self,
        *,
        name: str,
        role: str,
        my_gpu: GPU,
        replica_gpus: list[GPU],
        endpoint_port: int,
        dist_port: int,
        bind_host: str = "0.0.0.0",
        is_colocate: bool = False,
        runtime: RuntimeDir,
    ) -> None:
        super().__init__(name=name, role=role)
        self.my_gpu = my_gpu
        self.replica_gpus = list(replica_gpus)
        self.endpoint_port = endpoint_port
        self.dist_port = dist_port
        self.bind_host = bind_host
        self.is_colocate = is_colocate
        self.runtime = runtime

        self.master_gpu = self.replica_gpus[0]
        self.host = self.master_gpu.host
        ranks = [g.global_rank for g in self.replica_gpus]
        self.rank_in_replica = ranks.index(my_gpu.global_rank)
        self.is_master = self.rank_in_replica == 0

        self.engine: "SpmdEngine | None" = None
        self.worker: "TQWorker | None" = None
        self._readiness_event = None

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.endpoint_port}"

    # ── role hooks (run in the CHILD process) ────────────────────────────
    def build_engine(self) -> "SpmdEngine":  # pragma: no cover - interface
        """Construct this role's Engine. Runs in the engine subprocess."""
        raise NotImplementedError

    def build_worker(self, engine: "SpmdEngine", colocation) -> "TQWorker":
        """Construct this role's Worker around the initialized Engine."""
        raise NotImplementedError  # pragma: no cover - interface

    # ── lifecycle ────────────────────────────────────────────────────────
    def ignite(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._readiness_event = ctx.Event()
        p = ctx.Process(target=self._run_runtime)
        p.start()
        self.processes.append(p)

    def _run_runtime(self) -> None:
        """Child-process entry: build the Engine, hand it to the Worker, run."""
        log_path = self.runtime.process_log_path(self.name, self.rank_in_replica)
        redirect_output(log_path)
        logger.info("{} {} rank {} started; output redirected to {}", self.role, self.name, self.rank_in_replica, log_path)
        die_with_parent()
        self.engine = self.build_engine()
        self.engine.init()

        colocation = self.build_colocation_manager(
            on_acquire=self.engine.on_colocate_acquire,
            on_release=self.engine.on_colocate_release,
        )
        # Colocation is owned by rank 0. Its acquire/release callbacks dispatch
        # an SPMD command so every rank changes GPU residency symmetrically.
        if self.engine.is_master:
            colocation.start()

        self.worker = self.build_worker(self.engine, colocation)

        if not self.engine.is_master:
            self.engine.run_command_loop()
            return

        self.worker.tq_thread(name=f"{self.name}-tq-worker")
        assert self._readiness_event is not None
        self._readiness_event.set()
        self.engine.run_command_loop()

    def wait_for_ready(self, timeout: float = 3600.0) -> None:
        if not self.is_master:
            return
        import time

        if self._readiness_event is None:
            raise RuntimeError(f"{self.role} {self.name} readiness event was not created")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.role} {self.name} not ready within {timeout}s")
            if self._readiness_event.wait(timeout=min(1.0, remaining)):
                break
            for proc in self.processes:
                if not proc.is_alive():
                    raise RuntimeError(f"{self.role} {self.name} engine exited before ready")

        self.runtime.mark_ready(self.name)
        logger.info("{} {} ready; marker published", self.role, self.name)
