"""Common lifecycle primitives for Meshy workers."""

from __future__ import annotations

import threading
from typing import Any


class Worker:
    """Small synchronous lifecycle contract shared by role workers."""

    def __init__(self) -> None:
        self._worker_stop = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None

    @property
    def stopped(self) -> bool:
        return self._worker_stop.is_set()

    @property
    def worker_error(self) -> BaseException | None:
        return self._worker_error

    def start(self, *, name: str | None = None) -> threading.Thread:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            raise RuntimeError("worker is already running")
        self._worker_thread = threading.Thread(
            target=self._run_guarded,
            name=name or type(self).__name__,
            daemon=True,
        )
        self._worker_thread.start()
        return self._worker_thread

    def _run_guarded(self) -> None:
        try:
            self.run()
        except BaseException as exc:  # surfaced to the owning Service
            self._worker_error = exc
            raise

    def run(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:
        self._worker_stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._worker_thread is not None:
            self._worker_thread.join(timeout)

    def health_info(self) -> dict[str, Any]:
        return {
            "running": self._worker_thread is not None and self._worker_thread.is_alive(),
            "stopped": self.stopped,
            "error": str(self._worker_error) if self._worker_error else None,
        }
