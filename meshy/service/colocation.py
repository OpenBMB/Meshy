"""Request-ledger based decentralized GPU ownership for Meshy Services."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, NamedTuple

from loguru import logger

from meshy.transferqueue.colocation import (
    RequestHandle,
    RequestLedgerTransport,
    RequestRecord,
    TQRequestLedgerTransport,
)


class SchedulingMode(str, Enum):
    ON_DEMAND = "on_demand"
    FALLBACK = "fallback"


class RequestKind(str, Enum):
    ON_DEMAND = "on_demand"
    FALLBACK = "fallback"


class RingNode(NamedTuple):
    service_id: str
    mode: SchedulingMode


@dataclass(frozen=True)
class ColocationRing:
    group_id: str
    ring: tuple[RingNode, ...]
    poll_interval: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ring", tuple(RingNode(node[0], SchedulingMode(node[1])) for node in self.ring))
        if not self.group_id:
            raise ValueError("ColocationRing.group_id must be non-empty")
        if len(self.ring) < 2:
            raise ValueError("a colocation ring needs at least two Services")
        names = [node.service_id for node in self.ring]
        if any(not name for name in names):
            raise ValueError("colocation ring service ids must be non-empty")
        if len(names) != len(set(names)):
            raise ValueError(f"colocation ring contains duplicates: {names}")
        if self.poll_interval <= 0:
            raise ValueError("colocation poll_interval must be positive")

    @property
    def initial_owner(self) -> str:
        return self.ring[0].service_id

    def ring_index_for(self, service_id: str) -> int:
        for index, node in enumerate(self.ring):
            if node.service_id == service_id:
                return index
        raise KeyError(f"service {service_id!r} is not in ring {self.group_id!r}")

    def mode_for(self, service_id: str) -> SchedulingMode:
        return self.ring[self.ring_index_for(service_id)].mode


@dataclass(frozen=True)
class GpuRequest:
    group_id: str
    service_id: str
    request_id: str
    created_at_ns: int
    payload_ref: str | None = None
    ring_index: int = 0
    priority: int = 0
    kind: RequestKind = RequestKind.ON_DEMAND

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RequestKind(self.kind))


@dataclass(frozen=True)
class GpuGrant:
    group_id: str
    sequence: int
    source: str
    target: str
    request_id: str | None = None
    transition: str = ""
    payload_ref: str | None = None


@dataclass
class _ManagerCommand:
    action: str
    payload: Any
    done: threading.Event
    result: Any = None
    error: BaseException | None = None


def issue_genesis(config: ColocationRing, transport: RequestLedgerTransport) -> GpuGrant:
    """Create the initial owner through the same Request-row protocol."""
    request = GpuRequest(
        group_id=config.group_id,
        service_id=config.initial_owner,
        request_id=f"genesis:{config.group_id}",
        created_at_ns=0,
        ring_index=0,
        priority=1,
        kind=RequestKind.FALLBACK,
    )
    handle = transport.create_request(request)
    grant = GpuGrant(
        group_id=config.group_id,
        sequence=0,
        source="authority",
        target=config.initial_owner,
        request_id=request.request_id,
        transition="genesis",
    )
    transport.grant_request(handle, grant)
    return grant


class ColocationManager:
    """One Service's local view of a shared Request Ledger."""

    def __init__(
        self,
        config: ColocationRing,
        service_id: str,
        transport_factory: Callable[[], RequestLedgerTransport],
        *,
        on_acquire: Callable[[GpuGrant], None] | None = None,
        on_release: Callable[[str], None] | None = None,
    ) -> None:
        config.ring_index_for(service_id)
        self.config = config
        self.service_id = service_id
        self.mode = config.mode_for(service_id)
        self._transport_factory = transport_factory
        self._on_acquire = on_acquire or (lambda grant: None)
        self._on_release = on_release or (lambda target: None)
        self._commands: queue.Queue[_ManagerCommand] = queue.Queue()
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._fatal: BaseException | None = None
        self._records: dict[str, RequestRecord] = {}
        self._requests: dict[str, GpuRequest] = {}
        self._handles: dict[str, RequestHandle] = {}
        self._grants: dict[str, GpuGrant] = {}
        self._lease_events: dict[str, threading.Event] = {}
        self._acquired_sequences: set[int] = set()
        self._owner_service: str | None = None
        self._last_sequence = -1
        self._grant: GpuGrant | None = None

    @property
    def owns_gpu(self) -> bool:
        return self._owner_service == self.service_id

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ColocationManager is already started")
        self._thread = threading.Thread(
            target=self._run,
            name=f"colocate-{self.config.group_id}-{self.service_id}",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(30):
            raise TimeoutError(f"colocation manager {self.service_id} did not start")

    def request_gpu(
        self,
        request_id: str | None = None,
        payload_ref: str | None = None,
        *,
        kind: RequestKind | None = None,
    ) -> GpuRequest:
        request = GpuRequest(
            group_id=self.config.group_id,
            service_id=self.service_id,
            request_id=request_id or uuid.uuid4().hex,
            created_at_ns=time.time_ns(),
            payload_ref=payload_ref,
            ring_index=self.config.ring_index_for(self.service_id),
            priority=0 if (kind or RequestKind.ON_DEMAND) is RequestKind.ON_DEMAND else 1,
            kind=kind or RequestKind.ON_DEMAND,
        )
        self._submit("request", request)
        return request

    def wait_for_grant(self, request: GpuRequest, timeout: float | None = None) -> GpuGrant:
        event = self._lease_events.setdefault(request.request_id, threading.Event())
        if not event.wait(timeout):
            raise TimeoutError(f"timed out waiting for request {request.request_id!r}")
        if self._fatal is not None:
            raise RuntimeError("colocation manager failed") from self._fatal
        return self._grants[request.request_id]

    @contextmanager
    def occupy(self, request: GpuRequest, *, transition: str = "") -> Iterator[GpuGrant]:
        grant = self.wait_for_grant(request)
        try:
            yield grant
        finally:
            self.release(transition=transition)

    def release(self, *, transition: str = "", payload_ref: str | None = None) -> None:
        self._submit("release", (transition, payload_ref))

    def stop(self) -> None:
        if self._thread is None:
            return
        if self._thread.is_alive():
            self._submit("stop", None)
            self._thread.join(timeout=10)
        self._thread = None

    def _submit(self, action: str, payload: Any) -> Any:
        if self._thread is None:
            raise RuntimeError("ColocationManager is not started")
        if self._fatal is not None:
            raise RuntimeError("colocation manager failed") from self._fatal
        command = _ManagerCommand(action, payload, threading.Event())
        self._commands.put(command)
        if not command.done.wait(30):
            raise TimeoutError(f"colocation command {action!r} timed out")
        if command.error is not None:
            raise command.error
        return command.result

    def _run(self) -> None:
        transport = self._transport_factory()
        try:
            self._started.set()
            while not self._stop.is_set():
                self._drain_commands(transport)
                self._reconcile(transport.scan_requests(), transport)
                self._stop.wait(self.config.poll_interval)
        except BaseException as exc:  # noqa: BLE001
            self._fatal = exc
            logger.exception("Colocation manager {} failed", self.service_id)
            while True:
                try:
                    command = self._commands.get_nowait()
                except queue.Empty:
                    break
                command.error = exc
                command.done.set()
            for event in self._lease_events.values():
                event.set()
        finally:
            transport.close()

    def _drain_commands(self, transport: RequestLedgerTransport) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                if command.action == "request":
                    request = command.payload
                    handle = transport.create_request(request)
                    self._handles[request.request_id] = handle
                    self._requests[request.request_id] = request
                    command.result = request
                elif command.action == "release":
                    self._reconcile(transport.scan_requests(), transport)
                    transition, payload_ref = command.payload
                    command.result = self._transfer(
                        transport, transition=transition, payload_ref=payload_ref
                    )
                elif command.action == "stop":
                    self._stop.set()
                else:
                    raise ValueError(f"unknown colocation command {command.action!r}")
            except BaseException as exc:  # noqa: BLE001
                command.error = exc
            finally:
                command.done.set()

    def _reconcile(self, records: list[RequestRecord], transport: RequestLedgerTransport) -> None:
        records.sort(key=lambda record: record.grant.sequence if record.grant else -1)
        for record in records:
            request_id = record.handle.request.request_id
            previous = self._records.get(request_id)
            self._records[request_id] = record
            self._requests[request_id] = record.handle.request
            self._handles[request_id] = record.handle
            if record.state == "open" and self.owns_gpu and self.mode is SchedulingMode.FALLBACK:
                self._transfer(transport, transition="preempt")
                continue
            if record.grant is None:
                continue
            grant = record.grant
            if previous is not None and previous.grant is not None and previous.grant.sequence >= grant.sequence:
                continue
            self._apply_grant(grant)
            if self.mode is SchedulingMode.FALLBACK and not self.owns_gpu:
                self._ensure_fallback_request(transport)

    def _ensure_fallback_request(
        self, transport: RequestLedgerTransport, service_id: str | None = None
    ) -> None:
        service_id = service_id or self.service_id
        for record in self._records.values():
            req = record.handle.request
            if req.service_id == service_id and req.kind is RequestKind.FALLBACK and record.state == "open":
                return
        request = GpuRequest(
            group_id=self.config.group_id,
            service_id=service_id,
            request_id=f"{service_id}:fallback:{self._last_sequence + 1}",
            created_at_ns=time.time_ns(),
            ring_index=self.config.ring_index_for(service_id),
            priority=1,
            kind=RequestKind.FALLBACK,
        )
        handle = transport.create_request(request)
        self._handles[request.request_id] = handle
        self._requests[request.request_id] = request
        self._records[request.request_id] = RequestRecord(handle, "open", 0)

    def _apply_grant(self, grant: GpuGrant) -> None:
        if grant.group_id != self.config.group_id:
            return
        if grant.sequence <= self._last_sequence:
            return
        if self._last_sequence < 0:
            if grant.sequence != 0 or grant.source != "authority" or grant.target != self.config.initial_owner:
                raise RuntimeError(f"invalid genesis grant: {grant}")
        elif grant.sequence != self._last_sequence + 1 or self._grant is None or grant.source != self._grant.target:
            raise RuntimeError(f"non-contiguous GPU grant: last={self._grant}, next={grant}")
        self._last_sequence = grant.sequence
        self._grant = grant
        self._owner_service = grant.target
        if grant.request_id:
            self._grants[grant.request_id] = grant
        if grant.target == self.service_id and grant.sequence not in self._acquired_sequences:
            # The waiter must not proceed until the owner callback has
            # completed its SPMD residency command on every rank.
            self._on_acquire(grant)
            self._acquired_sequences.add(grant.sequence)
        if grant.request_id:
            self._lease_events.setdefault(grant.request_id, threading.Event()).set()

    def _select_next(self, transport: RequestLedgerTransport | None = None) -> RequestRecord | None:
        if self._owner_service != self.service_id:
            return None
        owner_index = self.config.ring_index_for(self.service_id)
        ring_size = len(self.config.ring)
        candidates = []
        for record in self._records.values():
            request = record.handle.request
            if record.state != "open" or record.grant is not None:
                continue
            if request.group_id != self.config.group_id or request.service_id == self.service_id:
                continue
            distance = (request.ring_index - owner_index) % ring_size
            candidates.append((request.priority, distance, request.created_at_ns, request.request_id, record))
        if not candidates:
            # A FALLBACK owner can be waiting for the token to return while
            # the ON_DEMAND owner releases immediately after its work. If the
            # source has not yet re-registered, create that return request at
            # the point of release and continue through the normal selector.
            if transport is not None and self._grant is not None and self._grant.source != self.service_id:
                self._ensure_fallback_request(transport, self._grant.source)
                return self._select_next(transport)
            return None
        candidates.sort(key=lambda item: item[:4])
        return candidates[0][4]

    def _transfer(
        self,
        transport: RequestLedgerTransport,
        *,
        transition: str,
        payload_ref: str | None = None,
    ) -> GpuGrant | None:
        if not self.owns_gpu:
            raise RuntimeError(f"service {self.service_id!r} does not own the GPU")
        selected = self._select_next(transport)
        if selected is None:
            return None
        self._on_release(selected.handle.request.service_id)
        current_request_id = self._grant.request_id if self._grant else None
        if current_request_id and current_request_id in self._handles:
            transport.close_request(self._handles[current_request_id])
        grant = GpuGrant(
            group_id=self.config.group_id,
            sequence=self._last_sequence + 1,
            source=self.service_id,
            target=selected.handle.request.service_id,
            request_id=selected.handle.request.request_id,
            transition=transition,
            payload_ref=payload_ref,
        )
        transport.grant_request(selected.handle, grant)
        selected.state = "granted"
        selected.grant = grant
        self._apply_grant(grant)
        logger.info(
            "Colocation GPU transfer: {} -> {} (sequence={}, request_id={}, transition={})",
            grant.source,
            grant.target,
            grant.sequence,
            grant.request_id,
            grant.transition,
        )
        return grant


class NoopColocationManager:
    owns_gpu = True

    def start(self) -> None: ...

    def request_gpu(self, request_id: str | None = None, payload_ref: str | None = None, **kwargs: Any) -> GpuRequest:
        return GpuRequest("", "", request_id or uuid.uuid4().hex, time.time_ns(), payload_ref)

    def wait_for_grant(self, request: GpuRequest, timeout: float | None = None) -> GpuGrant:
        return GpuGrant("", 0, "", "", request.request_id)

    @contextmanager
    def occupy(self, request: GpuRequest, **kwargs: Any) -> Iterator[GpuGrant]:
        yield self.wait_for_grant(request)

    def release(self, **kwargs: Any) -> None: ...

    def stop(self) -> None: ...


__all__ = [
    "ColocationRing",
    "ColocationManager",
    "GpuGrant",
    "GpuRequest",
    "NoopColocationManager",
    "RequestKind",
    "RingNode",
    "SchedulingMode",
    "issue_genesis",
]
