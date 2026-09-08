"""Shared runtime directory and store-backed readiness markers.

Endpoints / placement are **derived locally** from the SPMD-gathered GPU list
(see :mod:`meshy.service.topology`), so no cross-process *topology* files are
needed. The shared runtime directory (``XRL_RUNTIME_DIR``) now exists only for
on-disk artifacts: weight checkpoints ``weights/{name}/vN``, engine logs and
rollout trajectories.

Readiness markers -- "worker X is truly ready" (for colocate inference: only
after it has released its GPU memory) -- used to be empty files under
``ready/``; they now live in the run's bootstrap store
(:mod:`meshy.service.bootstrap`), so multi-node runs need no shared filesystem
for synchronization. The ``mark_ready`` / ``is_ready`` / ``wait_ready`` API is
unchanged; keys are namespaced by the runtime root because a single process
(pytest) may drive several runs back-to-back against one in-process store.
"""

from __future__ import annotations

import os

from meshy.service import bootstrap


class RuntimeDir:
    """Shared artifact directory + store-backed readiness-marker registry."""

    def __init__(
        self,
        root: str,
        *,
        checkpoint_dir: str | None = None,
        log_dir: str | None = None,
        tensorboard_dir: str | None = None,
        trajectory_dir: str | None = None,
    ) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.checkpoint_dir = os.path.abspath(
            checkpoint_dir
            or os.environ.get("XRL_CHECKPOINT_DIR")
            or os.path.join(self.root, "weights")
        )
        self.log_dir = os.path.abspath(
            log_dir or os.environ.get("XRL_LOG_DIR") or os.path.join(self.root, "logs")
        )
        self.tensorboard_dir = os.path.abspath(
            tensorboard_dir
            or os.environ.get("XRL_TENSORBOARD_DIR")
            or os.path.join(self.root, "tensorboard")
        )
        self.trajectory_dir = os.path.abspath(
            trajectory_dir
            or os.environ.get("XRL_TRAJECTORY_DIR")
            or self.root
        )

    @classmethod
    def from_env(cls, default_root: str | None = None) -> "RuntimeDir":
        root = os.environ.get("XRL_RUNTIME_DIR")
        if not root:
            root = default_root or os.path.join(".xrl_runtime", "default")
        return cls(root)

    # ── per-process artifacts ───────────────────────────────────────────
    def process_log_path(self, service_name: str, rank_in_service: int) -> str:
        """Return the append-only log for one service process.

        The service name keeps colocated roles separate while the rank within
        the service keeps multi-card SPMD processes separate.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        return os.path.join(self.log_dir, f"{service_name}-{int(rank_in_service)}.log")

    def checkpoint_path(self, service_name: str, version: int) -> str:
        """Return the HF checkpoint directory for one service version."""
        return os.path.join(self.checkpoint_dir, service_name, f"v{int(version)}")

    def tensorboard_path(self, service_name: str) -> str:
        """Return the TensorBoard event directory for one service."""
        return os.path.join(self.tensorboard_dir, service_name)

    def trajectory_path(self) -> str:
        """Return the JSONL trajectory log path for this run."""
        return os.path.join(self.trajectory_dir, "trajectories.jsonl")

    # ── readiness markers ────────────────────────────────────────────────
    def _ready_key(self, name: str) -> str:
        return f"ready|{self.root}|{name}"

    def mark_ready(self, name: str) -> None:
        """Publish ``name``'s readiness marker to the bootstrap store."""
        bootstrap.mark(self._ready_key(name))

    def is_ready(self, name: str) -> bool:
        return bootstrap.check([self._ready_key(name)])

    def wait_ready(
        self, names: list[str], timeout: float = 1800.0, interval: float = 1.0
    ) -> None:
        """Block until every worker in ``names`` has published its marker."""
        pending = list(dict.fromkeys(names))
        if not pending:
            return
        try:
            bootstrap.wait_keys(
                [self._ready_key(n) for n in pending], timeout=timeout, interval=interval
            )
        except TimeoutError:
            missing = [n for n in pending if not self.is_ready(n)]
            raise TimeoutError(
                f"readiness markers not published within {timeout}s: {missing} "
                f"(run {self.root})"
            ) from None
