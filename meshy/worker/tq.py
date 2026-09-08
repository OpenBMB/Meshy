"""TransferQueue contracts and worker-side data-flow helpers."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from loguru import logger

from meshy.worker.base import Worker


@dataclass(frozen=True)
class TQInput:
    partition: str
    fields: tuple[str, ...]
    batch_size: int
    consumer: str
    clear_after_success: bool = False

    def __post_init__(self) -> None:
        if not self.partition:
            raise ValueError("TQInput.partition must be non-empty")
        if not self.fields or any(not field for field in self.fields):
            raise ValueError("TQInput.fields must contain non-empty field names")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError(f"TQInput.fields contains duplicates: {self.fields}")
        if self.batch_size <= 0:
            raise ValueError(f"TQInput.batch_size must be positive, got {self.batch_size}")
        if not self.consumer:
            raise ValueError("TQInput.consumer must be non-empty")


@dataclass(frozen=True)
class TQOutput:
    """Output columns and whether the write creates new rows."""

    fields: tuple[str, ...]
    new_rows: bool
    partition: str | None = None

    def __post_init__(self) -> None:
        if not self.fields or any(not field for field in self.fields):
            raise ValueError("TQOutput.fields must contain non-empty field names")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError(f"TQOutput.fields contains duplicates: {self.fields}")
        if self.new_rows and not self.partition:
            raise ValueError("new-row TQOutput requires a partition")
        if not self.new_rows and self.partition is not None:
            raise ValueError("non-new-row TQOutput must not set partition")


class TQWorkerError(RuntimeError):
    """A fetched batch could not be completed within the retry budget."""


class TQWorker(Worker):
    """Unified TQ owner for both batch consumers and gate-driven producers.

    ``input`` is optional so a producer such as RolloutWorker can use the same
    client lifecycle and control polling without a separate SourceWorker type.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tq_input: TQInput | None = None
        self.tq_controls: dict[str, TQInput] = {}
        self.tq_outputs: dict[str, TQOutput] = {}
        self._tq_source_executor: ThreadPoolExecutor | None = None
        self._tq_source_client: Any | None = None

    def configure_tq(
        self,
        *,
        endpoints_ref: str,
        input: TQInput | None = None,
        controls: Mapping[str, TQInput] | None = None,
        outputs: Mapping[str, TQOutput] | None = None,
        poll_interval: float = 0.5,
        max_retries: int = 3,
        process_retries: int = 0,
        retry_interval: float = 1.0,
        paused: threading.Event | None = None,
        client_factory: Callable[[str], Any] | None = None,
        terminate_process_on_fatal: bool = False,
    ) -> None:
        if not hasattr(self, "_worker_stop"):
            Worker.__init__(self)
        if not endpoints_ref:
            raise ValueError("TQWorker requires a non-empty endpoints_ref")
        if max_retries < 0 or process_retries < 0:
            raise ValueError("TQWorker retry counts must be >= 0")
        controls = dict(controls or {})
        outputs = dict(outputs or {})
        if any(not name for name in controls) or any(not name for name in outputs):
            raise ValueError("TQWorker control/output names must be non-empty")
        self.tq_endpoints_ref = endpoints_ref
        self.tq_input = input
        self.tq_controls = controls
        self.tq_outputs = outputs
        self.tq_poll_interval = float(poll_interval)
        self.tq_max_retries = int(max_retries)
        self.tq_process_retries = int(process_retries)
        self.tq_retry_interval = float(retry_interval)
        self.tq_paused = paused
        self._tq_client_factory = client_factory
        self._tq_terminate_process_on_fatal = bool(terminate_process_on_fatal)
        self._tq_fatal_error: BaseException | None = None
        self._tq_stats = {
            "consumed": 0,
            "produced": 0,
            "batches": 0,
            "errors": 0,
            "process_seconds": 0.0,
        }

    # Backward-compatible spelling for callers that used the old migration API.
    configure_tq_worker = configure_tq

    def process_tq_batch(self, samples: list[Any]) -> Mapping[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def startup_tq_outputs(self) -> Mapping[str, Any]:
        return {}

    def _connect_tq(self) -> Any:
        if self._tq_client_factory is not None:
            return self._tq_client_factory(self.tq_endpoints_ref)
        from meshy.transferqueue import connect

        return connect(self.tq_endpoints_ref)

    def tq_thread(self, *, name: str | None = None) -> threading.Thread:
        return self.start(name=name or f"{type(self).__name__}-tq")

    def stop_tq_worker(self) -> None:
        self.stop()

    def tq_health_info(self) -> dict[str, Any]:
        return {
            "tq": dict(self._tq_stats),
            "tq_error": str(self._tq_fatal_error) if self._tq_fatal_error else None,
        }

    def on_tq_fatal(self, error: BaseException) -> None:
        self._tq_fatal_error = error
        logger.error("{} fatal error: {}", type(self).__name__, error)
        if self._tq_terminate_process_on_fatal:
            os.kill(os.getpid(), signal.SIGTERM)

    def run(self) -> None:
        self.run_tq_worker()

    def run_tq_worker(self) -> None:
        if self.tq_input is None:
            raise RuntimeError("run_tq_worker requires a primary TQ input")
        client = self._connect_tq()
        logger.info(
            "TQWorker: partition={!r} consumer={!r} batch={} fields={} outputs={}",
            self.tq_input.partition,
            self.tq_input.consumer,
            self.tq_input.batch_size,
            self.tq_input.fields,
            sorted(self.tq_outputs),
        )
        try:
            startup = self.startup_tq_outputs()
            self._retry_phase(
                "startup-write",
                lambda: self._write_outputs(client, None, startup),
                retries=self.tq_max_retries,
                batch_size=0,
            )
            while not self.stopped:
                if self.tq_paused is not None and self.tq_paused.is_set():
                    self._worker_stop.wait(self.tq_poll_interval)
                    continue
                worked = self.run_tq_once(client)
                if not worked:
                    self._worker_stop.wait(self.tq_poll_interval)
        except BaseException as exc:
            self.on_tq_fatal(exc)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def run_tq_once(self, client: Any) -> bool:
        if self.tq_input is None:
            raise RuntimeError("run_tq_once requires a primary TQ input")
        try:
            meta = client.get_meta(
                data_fields=list(self.tq_input.fields),
                batch_size=self.tq_input.batch_size,
                partition_id=self.tq_input.partition,
                mode="fetch",
                task_name=self.tq_input.consumer,
            )
        except Exception as exc:
            self._tq_stats["errors"] += 1
            logger.warning("TQWorker get_meta failed: {}", exc)
            return False
        if meta.size == 0:
            return False

        from meshy.transferqueue import adapter

        started = time.monotonic()
        packed = self._retry_phase(
            "read",
            lambda: client.get_data(meta),
            retries=self.tq_max_retries,
            batch_size=meta.size,
        )
        samples = adapter.td_to_samples(packed, self.tq_input.fields)
        result = self._retry_phase(
            "process",
            lambda: self.process_tq_batch(samples),
            retries=self.tq_process_retries,
            batch_size=meta.size,
        )
        self._retry_phase(
            "write",
            lambda: self._write_outputs(client, meta, result),
            retries=self.tq_max_retries,
            batch_size=meta.size,
        )
        if self.tq_input.clear_after_success:
            self._retry_phase(
                "clear",
                lambda: client.clear_samples(meta),
                retries=self.tq_max_retries,
                batch_size=meta.size,
            )
        self._tq_stats["consumed"] += int(meta.size)
        self._tq_stats["batches"] += 1
        self._tq_stats["process_seconds"] += time.monotonic() - started
        return True

    def _retry_phase(self, phase: str, fn: Callable[[], Any], *, retries: int, batch_size: int) -> Any:
        last_error: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                return fn()
            except BaseException as exc:
                last_error = exc
                self._tq_stats["errors"] += 1
                if attempt >= retries:
                    break
                logger.warning(
                    "{} {} phase failed for batch {} (attempt {}/{}), retrying: {}",
                    type(self).__name__, phase, batch_size, attempt + 1, retries + 1, exc,
                )
                if self._worker_stop.wait(self.tq_retry_interval):
                    break
        raise TQWorkerError(
            f"worker {self._tq_worker_name()!r} failed {phase} phase for batch of "
            f"{batch_size} after {retries + 1} attempt(s)"
        ) from last_error

    def _tq_worker_name(self) -> str:
        return self.tq_input.consumer if self.tq_input is not None else type(self).__name__

    def _write_outputs(self, client: Any, input_meta: Any | None, result: Mapping[str, Any]) -> None:
        if not isinstance(result, Mapping):
            raise TypeError(f"TQWorker result must be a mapping, got {type(result).__name__}")
        unknown = set(result) - set(self.tq_outputs)
        if unknown:
            raise ValueError(f"TQWorker returned undeclared outputs: {sorted(unknown)}")
        for name, data in result.items():
            spec = self.tq_outputs[name]
            actual_fields = set(data.keys())
            expected_fields = set(spec.fields)
            if actual_fields != expected_fields:
                raise ValueError(
                    f"output {name!r} fields mismatch: expected {sorted(expected_fields)}, got {sorted(actual_fields)}"
                )
            if spec.new_rows:
                client.put(data=data, partition_id=spec.partition)
            else:
                if input_meta is None:
                    raise ValueError(f"output {name!r} cannot modify rows without input metadata")
                if int(data.batch_size[0]) != int(input_meta.size):
                    raise ValueError(
                        f"output {name!r} has batch {int(data.batch_size[0])}, input has {int(input_meta.size)}"
                    )
                client.put(data=data, metadata=input_meta)
            self._tq_stats["produced"] += int(data.batch_size[0])

    # ---- shared async TQ access used by RolloutWorker -----------------
    async def open_tq(self) -> None:
        if self._tq_source_executor is not None:
            raise RuntimeError("TQ client is already open")
        self._tq_source_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tq-client")
        loop = asyncio.get_running_loop()
        self._tq_source_client = await loop.run_in_executor(self._tq_source_executor, self._connect_tq)

    @property
    def tq_executor(self) -> ThreadPoolExecutor:
        if self._tq_source_executor is None:
            raise RuntimeError("TQ client is not open")
        return self._tq_source_executor

    @property
    def tq_client(self) -> Any:
        if self._tq_source_client is None:
            raise RuntimeError("TQ client is not open")
        return self._tq_source_client

    async def tq_call(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self.tq_executor, fn, *args)

    async def write_tq_output(self, name: str, data: Any) -> None:
        if name not in self.tq_outputs:
            raise ValueError(f"TQWorker output {name!r} is not declared")
        started = time.monotonic()
        await self.tq_call(
            lambda: self._retry_phase(
                "write",
                lambda: self._write_outputs(self.tq_client, None, {name: data}),
                retries=self.tq_max_retries,
                batch_size=int(data.batch_size[0]),
            )
        )
        self._tq_stats["batches"] += 1
        self._tq_stats["process_seconds"] += time.monotonic() - started

    async def read_tq_control(self, name: str) -> Any | None:
        spec = self.tq_controls.get(name)
        if spec is None:
            raise ValueError(f"TQ control {name!r} is not declared")

        def read() -> Any | None:
            meta = self.tq_client.get_meta(
                data_fields=list(spec.fields),
                batch_size=spec.batch_size,
                partition_id=spec.partition,
                mode="fetch",
                task_name=spec.consumer,
            )
            if meta.size == 0:
                return None
            data = self.tq_client.get_data(meta)
            if spec.clear_after_success:
                self.tq_client.clear_samples(meta)
            return data

        return await self.tq_call(read)

    async def close_tq(self) -> None:
        executor = self._tq_source_executor
        client = self._tq_source_client
        if executor is None:
            return
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                await asyncio.get_running_loop().run_in_executor(executor, close)
        executor.shutdown(wait=True)
        self._tq_source_client = None
        self._tq_source_executor = None

    def health_info(self) -> dict[str, Any]:
        info = super().health_info()
        info.update(self.tq_health_info())
        return info


__all__ = ["TQInput", "TQOutput", "TQWorker", "TQWorkerError"]
