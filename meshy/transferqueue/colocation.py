"""TransferQueue request ledger used by Meshy's colocated GPU scheduler.

Each GPU request occupies one TQ row.  The request is inserted with empty
grant columns, then the current owner fills those columns using the original
row metadata.  Managers deliberately rescan the ledger with ``force_fetch``:
the grant is a mutation of an existing row, not a second append-only event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from transfer_queue.metadata import BatchMeta


REQUEST_ID = "request_id"
REQUEST_GROUP_ID = "request_group_id"
REQUEST_SERVICE_ID = "request_service_id"
REQUEST_RING_INDEX = "request_ring_index"
REQUEST_CREATED_AT_NS = "request_created_at_ns"
REQUEST_PRIORITY = "request_priority"
REQUEST_KIND = "request_kind"
REQUEST_PAYLOAD_REF = "request_payload_ref"
REQUEST_STATE = "request_state"
REQUEST_VERSION = "request_version"
GRANT_SEQUENCE = "grant_sequence"
GRANT_SOURCE = "grant_source"
GRANT_TARGET = "grant_target"
GRANT_REQUEST_ID = "grant_request_id"
GRANT_TRANSITION = "grant_transition"
GRANT_PAYLOAD_REF = "grant_payload_ref"
GRANT_COMMITTED_AT_NS = "grant_committed_at_ns"

REQUEST_FIELDS = (
    REQUEST_ID,
    REQUEST_GROUP_ID,
    REQUEST_SERVICE_ID,
    REQUEST_RING_INDEX,
    REQUEST_CREATED_AT_NS,
    REQUEST_PRIORITY,
    REQUEST_KIND,
    REQUEST_PAYLOAD_REF,
    REQUEST_STATE,
    REQUEST_VERSION,
)
GRANT_FIELDS = (
    GRANT_SEQUENCE,
    GRANT_SOURCE,
    GRANT_TARGET,
    GRANT_REQUEST_ID,
    GRANT_TRANSITION,
    GRANT_PAYLOAD_REF,
    GRANT_COMMITTED_AT_NS,
)
LEDGER_FIELDS = REQUEST_FIELDS + GRANT_FIELDS


@dataclass
class RequestHandle:
    """A request plus the metadata needed to update its TQ row."""

    request: Any
    metadata: "BatchMeta"
    version: int
    grant: Any | None = None


@dataclass
class RequestRecord:
    """Decoded row returned by a ledger scan."""

    handle: RequestHandle
    state: str
    version: int
    grant: Any | None = None


class RequestLedgerTransport(Protocol):
    def create_request(self, request: Any) -> RequestHandle: ...

    def scan_requests(self) -> list[RequestRecord]: ...

    def grant_request(self, handle: RequestHandle, grant: Any) -> None: ...

    def close_request(self, handle: RequestHandle) -> None: ...

    def purge_request(self, handle: RequestHandle) -> None: ...

    def close(self) -> None: ...


def _nontensor(value: Any) -> Any:
    """Unwrap ``NonTensorData``/``NonTensorStack`` values returned by TQ."""
    if hasattr(value, "data"):
        return value.data
    return value


def _scalar(value: Any) -> int:
    return int(value.reshape(-1)[0].item())


class TQRequestLedgerTransport:
    """TQ implementation of :class:`RequestLedgerTransport`.

    The transport owns one client and is intended to be called by one Manager
    thread.  ``force_fetch`` is intentional: row updates must remain visible
    after a manager has already observed the request once.
    """

    PARTITION_PREFIX = "control.colocate."
    PARTITION_SUFFIX = ".request"

    def __init__(self, endpoints_ref: str, group_id: str) -> None:
        from meshy.transferqueue.client import connect

        self.client = connect(endpoints_ref)
        self.partition = f"{self.PARTITION_PREFIX}{group_id}{self.PARTITION_SUFFIX}"
        # Create the partition without consuming its pre-allocated empty slot.
        # A force-fetch against a missing partition crashes the upstream
        # controller request thread, so every manager must establish it before
        # polling; batch_size=0 leaves the slot available for the first row.
        self.client.get_meta(
            data_fields=list(LEDGER_FIELDS),
            batch_size=0,
            partition_id=self.partition,
            mode="insert",
        )

    @staticmethod
    def _object_column(values: list[Any]):
        from tensordict import NonTensorData, NonTensorStack

        return NonTensorStack(*[NonTensorData(value) for value in values])

    @classmethod
    def _row_tensor_dict(cls, request: Any, *, state: str, version: int, grant: Any | None = None):
        import torch
        from tensordict import TensorDict

        grant = grant
        return TensorDict(
            {
                REQUEST_ID: cls._object_column([request.request_id]),
                REQUEST_GROUP_ID: cls._object_column([request.group_id]),
                REQUEST_SERVICE_ID: cls._object_column([request.service_id]),
                REQUEST_RING_INDEX: torch.tensor([[int(request.ring_index)]], dtype=torch.int64),
                REQUEST_CREATED_AT_NS: torch.tensor([[int(request.created_at_ns)]], dtype=torch.int64),
                REQUEST_PRIORITY: torch.tensor([[int(request.priority)]], dtype=torch.int64),
                REQUEST_KIND: cls._object_column([request.kind.value if hasattr(request.kind, "value") else str(request.kind)]),
                REQUEST_PAYLOAD_REF: cls._object_column([request.payload_ref]),
                REQUEST_STATE: cls._object_column([state]),
                REQUEST_VERSION: torch.tensor([[int(version)]], dtype=torch.int64),
                GRANT_SEQUENCE: torch.tensor(
                    [[int(grant.sequence) if grant is not None else -1]], dtype=torch.int64
                ),
                GRANT_SOURCE: cls._object_column([grant.source if grant is not None else ""]),
                GRANT_TARGET: cls._object_column([grant.target if grant is not None else ""]),
                GRANT_REQUEST_ID: cls._object_column(
                    [grant.request_id if grant is not None else ""]
                ),
                GRANT_TRANSITION: cls._object_column(
                    [grant.transition if grant is not None else ""]
                ),
                GRANT_PAYLOAD_REF: cls._object_column(
                    [getattr(grant, "payload_ref", None) if grant is not None else None]
                ),
                GRANT_COMMITTED_AT_NS: torch.tensor(
                    [[time.time_ns() if grant is not None else 0]], dtype=torch.int64
                ),
            },
            batch_size=[1],
        )

    def create_request(self, request: Any) -> RequestHandle:
        data = self._row_tensor_dict(request, state="open", version=0)
        metadata = self.client.put(data=data, partition_id=self.partition)
        return RequestHandle(request=request, metadata=metadata, version=0)

    def scan_requests(self) -> list[RequestRecord]:
        try:
            metadata = self.client.get_meta(
                data_fields=list(LEDGER_FIELDS),
                batch_size=1,
                partition_id=self.partition,
                mode="force_fetch",
            )
        except (RuntimeError, TimeoutError, ValueError):
            return []
        if metadata.size == 0:
            return []
        records: list[RequestRecord] = []
        # ``force_fetch`` includes the controller's pre-allocated but not yet
        # produced slots. Only rows whose requested fields are fully produced
        # can be decoded; this also avoids touching an empty storage key.
        active_positions = [
            index for index, status in enumerate(metadata.production_status) if int(status) == 1
        ]
        for index in active_positions:
            row_metadata = metadata.select_samples([index])
            try:
                data = self.client.get_data(row_metadata)
            except (RuntimeError, TimeoutError):
                break
            if REQUEST_ID not in data.keys():
                break
            request_id = str(_nontensor(data[REQUEST_ID][0]))
            group_id = str(_nontensor(data[REQUEST_GROUP_ID][0]))
            service_id = str(_nontensor(data[REQUEST_SERVICE_ID][0]))
            from meshy.service.colocation import GpuGrant, GpuRequest, RequestKind

            request = GpuRequest(
                group_id=group_id,
                service_id=service_id,
                request_id=request_id,
                created_at_ns=_scalar(data[REQUEST_CREATED_AT_NS][0]),
                payload_ref=_nontensor(data[REQUEST_PAYLOAD_REF][0]),
                ring_index=_scalar(data[REQUEST_RING_INDEX][0]),
                priority=_scalar(data[REQUEST_PRIORITY][0]),
                kind=RequestKind(str(_nontensor(data[REQUEST_KIND][0]))),
            )
            state = str(_nontensor(data[REQUEST_STATE][0]))
            version = _scalar(data[REQUEST_VERSION][0])
            grant = None
            sequence = _scalar(data[GRANT_SEQUENCE][0])
            if sequence >= 0:
                payload_value = data.get(GRANT_PAYLOAD_REF)
                grant = GpuGrant(
                    group_id=group_id,
                    sequence=sequence,
                    source=str(_nontensor(data[GRANT_SOURCE][0])),
                    target=str(_nontensor(data[GRANT_TARGET][0])),
                    request_id=str(_nontensor(data[GRANT_REQUEST_ID][0])) or None,
                    transition=str(_nontensor(data[GRANT_TRANSITION][0])),
                    payload_ref=(
                        _nontensor(payload_value[0])
                        if payload_value is not None
                        else None
                    ),
                )
            records.append(
                RequestRecord(
                    handle=RequestHandle(
                        request=request,
                        metadata=row_metadata,
                        version=version,
                        grant=grant,
                    ),
                    state=state,
                    version=version,
                    grant=grant,
                )
            )
        return records

    def grant_request(self, handle: RequestHandle, grant: Any) -> None:
        data = self._row_tensor_dict(
            handle.request,
            state="granted",
            version=handle.version + 1,
            grant=grant,
        )
        self.client.put(data=data, metadata=handle.metadata)
        handle.version += 1
        handle.grant = grant

    def close_request(self, handle: RequestHandle) -> None:
        data = self._row_tensor_dict(
            handle.request,
            state="closed",
            version=handle.version + 1,
            grant=handle.grant,
        )
        self.client.put(data=data, metadata=handle.metadata)
        handle.version += 1

    def purge_request(self, handle: RequestHandle) -> None:
        self.client.clear_samples(handle.metadata)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


__all__ = [
    "GRANT_FIELDS",
    "LEDGER_FIELDS",
    "REQUEST_FIELDS",
    "RequestHandle",
    "RequestLedgerTransport",
    "RequestRecord",
    "TQRequestLedgerTransport",
]
