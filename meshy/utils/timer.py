"""Lightweight timing utilities for RL pipeline instrumentation.

Two abstractions:

* :class:`TimerStats` — an accumulating, named-stopwatch registry. Use
  :meth:`TimerStats.timer` (context manager) to time a code block; the
  elapsed seconds get *added* into a per-name bucket. Use
  :meth:`TimerStats.pop_metrics` at the end of a logical iteration
  (e.g. one RL step) to flush the buckets into a flat metrics dict
  ready for wandb / loguru.
* :func:`log_timer` — a one-shot context manager that just logs elapsed
  time via loguru. Handy for one-off setup phases (warmup, eval) where
  there is no enclosing iteration to flush into.

Both honour an optional ``sync`` flag that issues
``torch.cuda.synchronize()`` at entry/exit so the measured wall time
reflects the GPU work (without sync, an async kernel launch returns
near-instantly and the stopwatch is meaningless for GPU-bound code).

Both also honour an ``enabled`` switch: when False, every operation
becomes a true no-op — no ``perf_counter`` reads, no
``cuda.synchronize`` calls, no bucket allocation, no log emission.
This lets recipes turn off all timing instrumentation (and the
non-trivial CUDA sync overhead it implies) via a single config flag
without touching call sites.

Accumulating semantics matter for inner loops: in a PPO ``train_step``
the same ``time/forward`` bucket fires once per microbatch; popping
once per outer step gives total forward time per step, which is what
you actually want on the dashboard.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch
from loguru import logger


def _maybe_sync(sync: bool) -> None:
    """``torch.cuda.synchronize()`` iff ``sync`` and CUDA is initialised.

    The ``is_initialized`` guard avoids forcing context init in
    CPU-only runs (e.g. unit tests).
    """
    if sync and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class TimerStats:
    """Accumulating named-stopwatch registry.

    Each call to :meth:`timer` adds the elapsed seconds into a bucket
    keyed by ``name``. Buckets stay until :meth:`pop_metrics` (flush
    + clear) or :meth:`reset` is called. ``count`` tracks how many
    times each name fired so callers can derive averages if desired.

    Set ``enabled=False`` to turn the whole registry into a no-op:
    :meth:`timer` skips ``perf_counter`` and ``cuda.synchronize``,
    :meth:`record` discards inputs, and all bucket dicts stay empty
    (so :meth:`pop_metrics` returns ``{}``). This is the master
    switch wired up via the recipe ``timer.enabled`` config flag.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._times: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @contextmanager
    def timer(self, name: str, *, sync: bool = False) -> Iterator[None]:
        """Time a block; add elapsed seconds into the ``name`` bucket.

        Args:
            name: Bucket key (no prefix; :meth:`pop_metrics` adds it).
            sync: If True, ``torch.cuda.synchronize()`` brackets the
                block so the measurement reflects GPU work. Use for any
                block that launches CUDA kernels you actually care
                about timing — without it you measure kernel launch
                latency, not kernel run time.

        When ``self.enabled`` is False this becomes a pure ``yield`` —
        no sync, no timing, no bucket write — so call sites pay zero
        overhead for instrumentation that's been turned off globally.
        """
        if not self.enabled:
            yield
            return
        _maybe_sync(sync)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _maybe_sync(sync)
            elapsed = time.perf_counter() - t0
            self._times[name] = self._times.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    def record(self, name: str, elapsed: float) -> None:
        """Manually add an externally-measured duration to a bucket."""
        if not self.enabled:
            return
        self._times[name] = self._times.get(name, 0.0) + float(elapsed)
        self._counts[name] = self._counts.get(name, 0) + 1

    def pop_metrics(self, prefix: str = "time/") -> dict[str, float]:
        """Flush all buckets into ``{prefix + name: seconds}`` and clear.

        Counts are reset alongside times. Returns ``{}`` when disabled
        (buckets are never populated, so this is just a fast path).
        """
        if not self.enabled:
            return {}
        out = {f"{prefix}{k}": v for k, v in self._times.items()}
        self._times.clear()
        self._counts.clear()
        return out

    def snapshot(self, prefix: str = "time/") -> dict[str, float]:
        """Read current buckets without clearing them."""
        if not self.enabled:
            return {}
        return {f"{prefix}{k}": v for k, v in self._times.items()}

    def reset(self) -> None:
        self._times.clear()
        self._counts.clear()

    def __bool__(self) -> bool:
        return bool(self._times)


@contextmanager
def log_timer(
    name: str,
    *,
    sync: bool = False,
    level: str = "INFO",
    enabled: bool = True,
) -> Iterator[None]:
    """One-shot timer that logs elapsed seconds via loguru on exit.

    Use for ad-hoc phases (warmup, suite eval, checkpoint save) where
    there is no enclosing iteration to aggregate into. For per-step
    timing inside the train loop, use :class:`TimerStats` instead.

    Pass ``enabled=False`` to skip all measurement *and* the log line
    entirely — the block runs as a plain ``yield``. Wired up to the
    recipe ``timer.enabled`` config flag.
    """
    if not enabled:
        yield
        return
    _maybe_sync(sync)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _maybe_sync(sync)
        elapsed = time.perf_counter() - t0
        logger.log(level, "[timer] {} took {:.3f}s", name, elapsed)


def format_timer_summary(metrics: dict[str, float], prefix: str = "time/") -> str:
    """Pretty-print timer metrics for loguru, e.g. ``rollout=1.23s ...``.

    Strips ``prefix`` from each key and sorts longest-first so the
    expensive stages stand out at a glance.
    """
    items = [
        (k[len(prefix):] if k.startswith(prefix) else k, v)
        for k, v in metrics.items()
    ]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return "  ".join(f"{k}={v:.3f}s" for k, v in items)
