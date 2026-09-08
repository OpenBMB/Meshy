from __future__ import annotations

import threading
import time

from meshy.service.colocation import (
    ColocationRing,
    ColocationManager,
    GpuGrant,
    RequestKind,
    SchedulingMode,
    issue_genesis,
)
from meshy.transferqueue.colocation import RequestHandle, RequestRecord


class MemoryLedger:
    def __init__(self) -> None:
        self.rows: dict[str, list] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def create_request(self, request):
        with self._lock:
            self._counter += 1
            handle = RequestHandle(request, self._counter, 0)
            self.rows[request.request_id] = [handle, "open", 0, None]
            return handle

    def scan_requests(self):
        with self._lock:
            return [RequestRecord(handle, state, version, grant) for handle, state, version, grant in self.rows.values()]

    def grant_request(self, handle, grant: GpuGrant) -> None:
        with self._lock:
            row = self.rows[handle.request.request_id]
            row[1] = "granted"
            row[2] += 1
            row[3] = grant
            handle.version = row[2]

    def close_request(self, handle) -> None:
        with self._lock:
            row = self.rows[handle.request.request_id]
            row[1] = "closed"
            row[2] += 1
            handle.version = row[2]

    def purge_request(self, handle) -> None:
        with self._lock:
            self.rows.pop(handle.request.request_id, None)

    def close(self) -> None:
        pass


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        time.sleep(0.001)


def test_request_row_lifecycle_and_ring_priority() -> None:
    config = ColocationRing(
        "cards",
        (
            ("rollout", SchedulingMode.FALLBACK),
            ("titan", SchedulingMode.ON_DEMAND),
        ),
        poll_interval=0.001,
    )
    ledger = MemoryLedger()
    rollout = ColocationManager(config, "rollout", lambda: ledger)
    titan = ColocationManager(config, "titan", lambda: ledger)
    rollout.start()
    titan.start()
    try:
        issue_genesis(config, ledger)
        wait_until(lambda: rollout.owns_gpu)

        request = titan.request_gpu("titan:step:1")
        wait_until(lambda: titan.owns_gpu)
        assert titan.wait_for_grant(request) == titan._grants[request.request_id]
        assert ledger.rows[request.request_id][1] == "granted"

        fallback = rollout.request_gpu("rollout:return:1", kind=RequestKind.FALLBACK)
        titan.release(transition="step-complete", payload_ref="/run/weights/v1")
        wait_until(lambda: rollout.owns_gpu)
        assert titan.owns_gpu is False
        assert ledger.rows[request.request_id][1] == "closed"
        assert ledger.rows[fallback.request_id][3].target == "rollout"
        assert ledger.rows[fallback.request_id][3].payload_ref == "/run/weights/v1"
    finally:
        rollout.stop()
        titan.stop()


def test_on_demand_request_beats_fallback_request() -> None:
    config = ColocationRing(
        "cards",
        (
            ("rollout", SchedulingMode.FALLBACK),
            ("teacher", SchedulingMode.ON_DEMAND),
            ("titan", SchedulingMode.ON_DEMAND),
        ),
        poll_interval=0.001,
    )
    manager = ColocationManager(config, "rollout", lambda: MemoryLedger())
    manager._owner_service = "rollout"
    manager._grant = GpuGrant("cards", 0, "authority", "rollout")
    manager._last_sequence = 0
    # Build pending records directly to isolate deterministic ordering.
    for request_id, service_id, kind, created in (
        ("fallback", "rollout", RequestKind.FALLBACK, 1),
        ("teacher", "teacher", RequestKind.ON_DEMAND, 2),
        ("titan", "titan", RequestKind.ON_DEMAND, 3),
    ):
        from meshy.service.colocation import GpuRequest

        request = GpuRequest(
            "cards",
            service_id,
            request_id,
            created,
            ring_index=config.ring_index_for(service_id),
            priority=0 if kind is RequestKind.ON_DEMAND else 1,
            kind=kind,
        )
        handle = RequestHandle(request, object(), 0)
        manager._records[request_id] = RequestRecord(handle, "open", 0)

    assert manager._select_next().handle.request.service_id == "teacher"
