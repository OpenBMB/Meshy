"""Card-level SPMD Ignitor: GPU discovery, topology derivation, and startup.

Under card-level SPMD there is **one ignitor process per GPU card** (launched
via a single ``torchrun --nproc-per-node=<cards_per_node>``); the global
``RANK`` is the card index. Each ignitor process:

1. discovers its own card and all-gathers the global card list;
2. recomputes the deterministic full :class:`~meshy.service.topology.Topology`
   from the :class:`~meshy.service.base.ServiceGroup` DAG (endpoints included --
   no info files);
3. selects *all* Services assigned to its card, and for each (in dependency
   order) waits for its ``wait_until`` dependencies (readiness markers), then
   ``ignite()``s and ``wait_for_ready()``s it (which publishes its own marker).

Role dispatch is data-driven: the ignitor looks the role up in
:mod:`~meshy.service.registry` and calls the Service class's ``from_info``
factory. All role-specific wiring (colocate hand-off endpoints, weight-sync
targets, ...) lives inside each Service's ``from_info``, so the ignitor knows
nothing about individual roles.

Colocate (inference + trainer time-sharing the same cards) needs **no** second
torchrun: the one ignitor process on each shared card ignites the inference
replica first (which releases its GPU memory and publishes its readiness marker
on ready), then the colocate trainer, whose ``wait_until`` gate blocks on those
inference markers before it loads onto the freed cards. SGLang still runs as a
subprocess (see :mod:`meshy.service.inference`) so that its own
``torch.distributed`` init stays out of the ignitor's torchrun rendezvous -- but
that is a process-isolation concern, not an environment one, and it needs no
separate torchrun.
"""

from __future__ import annotations

import os
import socket
import time

from loguru import logger

from meshy.service.base import GPU, Service, ServiceGroup, redirect_output
from meshy.service.registry import resolve_service
from meshy.service.runtime import RuntimeDir
from meshy.service.topology import ServiceInfo, Topology, build_topology

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meshy.service.colocation import ColocationRing


# ── GPU discovery ────────────────────────────────────────────────────────────
def _discover_host() -> str:
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world == 1:
        return "127.0.0.1"
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", 29500))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((master, port))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def _discover_gpus() -> list[GPU]:
    # NOTE: deliberately does NOT touch ``torch.cuda`` — the ignitor must not
    # create a CUDA context (it would pin memory on the very card that colocate
    # hands back and forth between the inference and trainer engines).
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    node = int(os.environ.get("GROUP_RANK", 0))
    me = GPU(host=_discover_host(), global_rank=rank, node_rank=node, local_rank=local)
    if world == 1:
        return [me]
    import torch.distributed as dist

    gathered: list = [None] * world
    dist.all_gather_object(gathered, me)
    return sorted(gathered, key=lambda g: g.global_rank)


def _gpu_by_rank(gpus: list[GPU], rank: int) -> GPU:
    for g in gpus:
        if g.global_rank == rank:
            return g
    raise KeyError(f"no GPU discovered for rank {rank}")


# ── Ignitor ──────────────────────────────────────────────────────────────────
class Ignitor:
    """Orchestrates one card-level SPMD pass over a :class:`ServiceGroup` list."""

    def __init__(
        self,
        service_groups: list[ServiceGroup],
        colocations: list["ColocationRing"] | None = None,
    ) -> None:
        self.service_groups = list(service_groups)
        self.colocations = list(colocations or [])
        self.services: list[Service] = []
        self.runtime: RuntimeDir | None = None

    def _resolve_runtime(self) -> RuntimeDir:
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        root = os.environ.get("XRL_RUNTIME_DIR")
        if not root:
            root = (
                os.path.abspath(os.path.join(".xrl_runtime", time.strftime("%Y%m%d-%H%M%S")))
                if rank == 0
                else None
            )
            if world > 1:
                import torch.distributed as dist

                container = [root]
                dist.broadcast_object_list(container, src=0)
                root = container[0]
        os.environ["XRL_RUNTIME_DIR"] = root
        return RuntimeDir(root)

    def run(self) -> None:
        world = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        if world > 1:
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(backend="gloo")

        self.runtime = self._resolve_runtime()
        # Keep Ignitor diagnostics separate by global card rank. Role child
        # processes switch to their own service/rank log when they start.
        log_path = self.runtime.process_log_path("ignitor", rank)
        redirect_output(log_path)
        logger.info("Ignitor rank {} started; output redirected to {}", rank, log_path)
        # Resolve the bootstrap store before igniting anything: under a direct
        # ``torchrun -m recipe.x`` (no launcher) this makes rank 0 host it, and
        # in every mode it exports XRL_BOOTSTRAP_ADDR so engine subprocesses
        # and the rollout driver connect as clients.
        from meshy.service import bootstrap

        bootstrap.get_store()
        gpus = _discover_gpus()
        topology = build_topology(self.service_groups, gpus, self.colocations)
        if rank == 0:
            logger.info(
                "Ignitor: {} cards discovered, runtime={}",
                len(gpus),
                self.runtime.root,
            )

        # Ignite every GPU service assigned to this card, in dependency order:
        # for a colocate card that is the inference replica (which frees its GPU
        # memory on ready) followed by the trainer sharing the card, whose
        # ``wait_until`` gate blocks on the inference marker just published here.
        my_gpu = _gpu_by_rank(gpus, rank)
        my_infos = topology.local_services(rank)
        if not my_infos:
            logger.info("Rank {} has no GPU service", rank)
        for info in my_infos:
            service = self._build_service(info, my_gpu, topology)
            self._gate_and_start(service, info, topology)
            self.services.append(service)

        # Colocation managers are now online but no role owns the runtime token
        # until every GPU Service has completed its existing bootstrap handoff.
        if self.colocations:
            if world > 1:
                import torch.distributed as dist

                dist.barrier()
            if rank == 0:
                self._issue_colocation_genesis()

        # CPU-only Services are launched once, by global rank 0, after their
        # ``wait_until`` dependencies are ready via markers.
        if rank == 0:
            for cpu_info in topology.cpu_services():
                service = self._build_service(cpu_info, None, topology)
                self._gate_and_start(service, cpu_info, topology)
                self.services.append(service)

        if self.services:
            logger.info("Rank {} ready: {}", rank, [s.name for s in self.services])

        self.join()

    def _gate_and_start(self, service: Service, info: ServiceInfo, topology: Topology) -> None:
        """Wait for ``info``'s dependencies, ignite, then wait until ready.

        The readiness gate happens *before* ``ignite()`` so a colocate trainer
        never loads onto a card until the inference sharing it has freed GPU
        memory (published its marker). ``wait_for_ready`` then publishes this
        service's own marker, unblocking whatever depends on it.
        """
        assert self.runtime is not None
        deps = topology.dependency_names(info)
        if deps:
            logger.info("Service {} waiting for readiness of {}", info.name, deps)
            self.runtime.wait_ready(deps)
        logger.info("Igniting service {} ({})", info.name, info.role)
        service.ignite()
        service.wait_for_ready()

    def _build_service(
        self, info: ServiceInfo, my_gpu: GPU | None, topology: Topology
    ) -> Service:
        if info.config is None:
            raise ValueError(f"service {info.name!r} has no ServiceConfig")
        service = resolve_service(info.config).from_info(
            info, my_gpu, topology, self.runtime
        )
        service.configure_colocation(info, self.runtime)
        return service

    def _issue_colocation_genesis(self) -> None:
        assert self.runtime is not None
        from meshy.transferqueue.colocation import TQRequestLedgerTransport
        from meshy.service.colocation import issue_genesis
        from meshy.transferqueue.client import resolve_endpoints_file

        endpoints_ref = resolve_endpoints_file(self.runtime.root)
        for config in self.colocations:
            transport = TQRequestLedgerTransport(endpoints_ref, config.group_id)
            try:
                issue_genesis(config, transport)
            finally:
                transport.close()

    def join(self) -> None:
        procs = [p for s in self.services for p in s.processes]
        if not procs:
            # Passive / idle ranks (e.g. non-master inference cards, or cards
            # not used by this pass) keep the torchrun group uniform by idling.
            logger.info("Rank {} has no local engine process; idling.", os.environ.get("RANK", 0))
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                return
        for p in procs:
            try:
                p.join()
            except KeyboardInterrupt:
                break
