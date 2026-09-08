"""Deterministic, file-free topology derivation for the card-level SPMD stack.

Under card-level SPMD every ignitor process already all-gathers the global list
of :class:`~meshy.service.base.GPU` (host / global_rank / node_rank / local_rank).
Given that list plus the static :class:`~meshy.service.base.ServiceGroup` DAG,
every process can *locally* recompute the complete placement of every service --
which cards it occupies, on which host, and (deterministically) at which
endpoint / dist port. No cross-process info files are needed to discover
endpoints.

:func:`build_topology` is a pure function of ``(service_groups, gpus)``: it
yields the full registry of every :class:`ServiceInfo`. Which of those a given
card's ignitor process actually ignites is selected afterwards by
:meth:`Topology.local_services`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from meshy.config import ServiceConfig
from meshy.service.base import GPU, ServiceGroup

if TYPE_CHECKING:
    from meshy.service.colocation import ColocationRing

# Built-in port bases are retained as public compatibility constants. The
# topology itself reads bases from each ServiceConfig subclass, allowing custom
# Services to participate without adding another role branch here.
INFER_PORT_BASE = 30000
INFER_DIST_BASE = 40000
TRAIN_PORT_BASE = 31000
TRAIN_DIST_BASE = 41000


@dataclass
class ServiceInfo:
    """A single resolved replica: where it lives and how to reach it.

    Pure data (all fields are plain / dataclass values) so it can be pickled
    into a spawned engine subprocess. ``replica_gpus`` is empty for CPU-only
    Services, which have no endpoint. ``config`` is the group's typed
    :class:`~meshy.config.ServiceConfig`, carried whole so nothing a
    recipe sets can be dropped between the recipe and the engine.
    """

    name: str  # f"{group_id}-{replica_idx}"
    group_id: str
    role: str
    uses_gpu: bool
    replica_idx: int
    host: str
    endpoint_port: int
    dist_port: int
    replica_gpus: list[GPU]
    # NOT redundant with ``colocate_with``: the latter is a *directional* pointer
    # set only on the side that reuses another group's cards (e.g. the trainer).
    # ``is_colocate`` is the *symmetric* "participates in a colocate arrangement"
    # flag -- true for BOTH sides, including the group being colocated onto (e.g.
    # inference, whose ``colocate_with`` is None yet must release GPU memory on
    # ready). i.e. ``is_colocate = (colocate_with is not None) or (someone
    # colocates onto me)``. A standalone service carries no topology-wide view,
    # so this must be resolved at build time and travel with the ServiceInfo.
    is_colocate: bool
    colocate_with: str | None
    wait_until: list[str] = field(default_factory=list)
    config: ServiceConfig | None = None
    colocation: "ColocationRing | None" = None
    is_colocation_leader: bool = False

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.endpoint_port}"

    @property
    def master_rank(self) -> int:
        return self.replica_gpus[0].global_rank

    def contains_rank(self, rank: int) -> bool:
        return any(g.global_rank == rank for g in self.replica_gpus)


@dataclass
class Topology:
    """The full, pass-independent registry of every service replica."""

    services: list[ServiceInfo]

    # ── lookups ──────────────────────────────────────────────────────────
    def group(self, group_id: str) -> list[ServiceInfo]:
        return [s for s in self.services if s.group_id == group_id]

    def by_name(self, name: str) -> ServiceInfo:
        for s in self.services:
            if s.name == name:
                return s
        raise KeyError(f"no service named {name!r}")

    def gpu_services(self) -> list[ServiceInfo]:
        return [s for s in self.services if s.uses_gpu]

    def cpu_services(self) -> list[ServiceInfo]:
        return [s for s in self.services if not s.uses_gpu]

    def services_by_role(self, role: str) -> list[ServiceInfo]:
        return [s for s in self.services if s.role == role]

    def rollout_services(self) -> list[ServiceInfo]:
        return self.services_by_role("rollout")

    def inference_services(self) -> list[ServiceInfo]:
        return self.services_by_role("inference")

    def training_services(self) -> list[ServiceInfo]:
        return self.services_by_role("training")

    # ── derived endpoints (replaces file-based discovery) ────────────────
    def inference_endpoints(self) -> list[str]:
        return [s.endpoint for s in self.inference_services()]

    def inference_targets(self) -> list[dict[str, Any]]:
        """All inference replicas a trainer may push weights to (name+endpoint)."""
        return [{"name": s.name, "endpoint": s.endpoint} for s in self.inference_services()]

    def trainer_endpoint(self) -> str | None:
        trainers = self.training_services()
        return trainers[0].endpoint if trainers else None

    # ── readiness dependencies ───────────────────────────────────────────
    def dependency_names(self, service: ServiceInfo) -> list[str]:
        """Names of every service in ``service``'s ``wait_until`` groups."""
        names: list[str] = []
        for gid in service.wait_until:
            names.extend(s.name for s in self.group(gid))
        return names

    # ── local service selection ──────────────────────────────────────────
    def local_services(self, rank: int) -> list[ServiceInfo]:
        """Every GPU service this rank ignites, in dependency-safe order.

        A single ``torchrun`` launches one ignitor process per card, and that
        process ignites *all* GPU services assigned to its card. In a colocate
        arrangement a card carries two services -- the inference replica it
        hosts and the trainer sharing that card. They are returned in topology
        (declaration) order, which guarantees the "colocated onto" group
        (inference) precedes the colocating group (trainer): igniting in that
        order lets the trainer's ``wait_until`` gate observe the inference
        readiness marker this same process just published, instead of
        dead-locking on a marker it has not written yet. CPU-only Services are
        handled separately (rank 0) and never returned here.
        """
        return [s for s in self.gpu_services() if s.contains_rank(rank)]

    # ── colocate memory-handoff wiring for a trainer replica ─────────────
    def colocated_inference_for(
        self, trainer: ServiceInfo
    ) -> tuple[list[str], dict[int, str], list[str]]:
        """(names, by_rank, endpoints) of inference replicas sharing trainer cards.

        A colocate trainer's rank 0 releases / resumes / weight-syncs every
        inference replica sharing its card block; each trainer rank must also
        know the specific inference replica on *its* card. Returns the managed
        inference replica names, a ``global_rank -> inference name`` map, and the
        managed endpoints (deduped, order-stable).
        """
        if trainer.colocate_with is None:
            return [], {}, []
        my_ranks = {g.global_rank for g in trainer.replica_gpus}
        by_rank: dict[int, str] = {}
        names: list[str] = []
        endpoints: list[str] = []
        for inf in self.group(trainer.colocate_with):
            shared = [g.global_rank for g in inf.replica_gpus if g.global_rank in my_ranks]
            if not shared:
                continue
            for gr in shared:
                by_rank[gr] = inf.name
            names.append(inf.name)
            endpoints.append(inf.endpoint)
        return names, by_rank, endpoints

def build_topology(
    service_groups: list[ServiceGroup],
    gpus: list[GPU],
    colocations: list["ColocationRing"] | None = None,
) -> Topology:
    """Resolve every replica's placement + endpoint. Pure & pass-independent.

    Card assignment walks the groups in order over the sorted GPU list: a group
    without ``colocate_with`` claims the next ``n_replicas * n_gpus_per_replica``
    cards; a group with ``colocate_with=X`` reuses X's exact card block (X must
    have been declared earlier and occupy the same number of cards). The
    CPU-only Services claim no cards.
    """
    gpus = sorted(gpus, key=lambda g: g.global_rank)
    colocated_targets = {sg.colocate_with for sg in service_groups if sg.colocate_with}
    card_ranges: dict[str, list[GPU]] = {}
    services: list[ServiceInfo] = []
    cursor = 0

    for sg in service_groups:
        config_type = type(sg.config)
        if not sg.role or sg.role == "?":
            raise ValueError(
                f"ServiceConfig {config_type.__name__} must declare a role"
            )

        if not sg.uses_gpu:
            for r in range(sg.n_replicas):
                services.append(
                    ServiceInfo(
                        name=f"{sg.id}-{r}",
                        group_id=sg.id,
                        role=sg.role,
                        uses_gpu=False,
                        replica_idx=r,
                        host="127.0.0.1",
                        endpoint_port=0,
                        dist_port=0,
                        replica_gpus=[],
                        is_colocate=False,
                        colocate_with=None,
                        wait_until=list(sg.wait_until),
                        config=sg.config,
                    )
                )
            continue

        total = sg.n_replicas * sg.n_gpus_per_replica
        if sg.colocate_with is not None:
            if sg.colocate_with not in card_ranges:
                raise ValueError(
                    f"group {sg.id!r} colocate_with={sg.colocate_with!r} which is not a "
                    "GPU group declared earlier"
                )
            block = card_ranges[sg.colocate_with]
            if len(block) != total:
                raise ValueError(
                    f"colocate group {sg.id!r} occupies {total} cards but "
                    f"{sg.colocate_with!r} occupies {len(block)}; they must match"
                )
        else:
            block = gpus[cursor : cursor + total]
            if len(block) != total:
                raise ValueError(
                    f"group {sg.id!r} needs {total} cards but only {len(block)} remain"
                )
            cursor += total
            card_ranges[sg.id] = block

        is_colocate = sg.colocate_with is not None or sg.id in colocated_targets
        c = 0
        for r in range(sg.n_replicas):
            replica_gpus = block[c : c + sg.n_gpus_per_replica]
            c += sg.n_gpus_per_replica
            master_rank = replica_gpus[0].global_rank
            endpoint_base = config_type.endpoint_port_base
            dist_base = config_type.dist_port_base
            if endpoint_base is None or dist_base is None:
                raise ValueError(
                    f"GPU ServiceConfig {config_type.__name__} must declare "
                    "endpoint_port_base and dist_port_base"
                )
            endpoint_port = endpoint_base + master_rank
            dist_port = dist_base + master_rank
            services.append(
                ServiceInfo(
                    name=f"{sg.id}-{r}",
                    group_id=sg.id,
                    role=sg.role,
                    uses_gpu=True,
                    replica_idx=r,
                    host=replica_gpus[0].host,
                    endpoint_port=endpoint_port,
                    dist_port=dist_port,
                    replica_gpus=replica_gpus,
                    is_colocate=is_colocate,
                    colocate_with=sg.colocate_with,
                    wait_until=list(sg.wait_until),
                    config=sg.config,
                )
            )

    _attach_colocations(services, colocations or [])
    _validate_ports(services)
    return Topology(services)


def _attach_colocations(
    services: list[ServiceInfo],
    colocations: list["ColocationRing"],
) -> None:
    if not colocations:
        return
    config_ids = [config.group_id for config in colocations]
    if len(set(config_ids)) != len(config_ids):
        raise ValueError(f"duplicate colocation group ids: {config_ids}")
    known_groups = {service.group_id for service in services}
    assigned: set[str] = set()
    for config in colocations:
        ring_groups = {node.service_id for node in config.ring}
        missing = ring_groups - known_groups
        if missing:
            raise ValueError(
                f"colocation {config.group_id!r} references unknown groups: "
                f"{sorted(missing)}"
            )
        overlap = ring_groups & assigned
        if overlap:
            raise ValueError(
                f"ServiceGroups can belong to only one colocation ring: "
                f"{sorted(overlap)}"
            )
        assigned.update(ring_groups)
        card_sets = {
            group_id: {
                gpu.global_rank
                for service in services
                if service.group_id == group_id
                for gpu in service.replica_gpus
            }
            for group_id in ring_groups
        }
        if len({frozenset(cards) for cards in card_sets.values()}) != 1:
            raise ValueError(
                f"colocation ring {config.group_id!r} members do not share the "
                f"same GPU block: {card_sets}"
            )
        for group_id in ring_groups:
            members = [service for service in services if service.group_id == group_id]
            if not members or not all(service.is_colocate for service in members):
                raise ValueError(
                    f"colocation ring member {group_id!r} is not placed on "
                    "colocated GPUs"
                )
            # A ring member is a ServiceGroup and may have multiple replicas.
            # The group needs exactly one ledger manager, while every ring
            # member group needs its own manager so it can request/release the
            # token.  Elect replica 0 within each group rather than the first
            # service in the whole ring (which would strand later members when
            # an inference group has more than one replica).
            for service in members:
                service.colocation = config
                service.is_colocation_leader = service.replica_idx == 0


def _validate_ports(services: list[ServiceInfo]) -> None:
    """Reject ambiguous plugin port assignments before any process starts."""
    for attribute in ("endpoint_port", "dist_port"):
        owners: dict[tuple[str, int], str] = {}
        for service in services:
            if not service.uses_gpu:
                continue
            port = getattr(service, attribute)
            if not 1 <= port <= 65535:
                raise ValueError(
                    f"service {service.name!r} has invalid {attribute} {port}"
                )
            address = (service.host, port)
            previous = owners.get(address)
            if previous is not None:
                raise ValueError(
                    f"{attribute} collision on {service.host}:{port} between "
                    f"services {previous!r} and {service.name!r}; choose distinct "
                    "port bases on their ServiceConfig classes"
                )
            owners[address] = service.name
