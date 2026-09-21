"""Global checkpoint state store: shard layout, completion, resume resolution.

A Meshy run has no driver, so it has no single process that could write "the"
checkpoint. Instead each role persists **its own shard** at a shared clock --
the global window index -- and the run's resume point is derived, not recorded:

    resume_step = max{ N : every participant at N has a complete shard at N }

``participants(N)`` is a pure function of the recipe (see
:meth:`meshy.service.topology.Topology.checkpoint_participants`), so every
process computes the same ``resume_step`` from the same directory listing
without talking to anyone. That is the same trick the topology itself uses.

Layout under ``RuntimeDir.state_dir`` (``XRL_STATE_DIR`` or ``{root}/state``)::

    state/
      actor_train-0/step-30/            torchtitan DCP (model+optim+lr+step)
      actor_train-0/step-30.json        meta: counters, exported weights path
      actor_train-0/step-30.complete    written last; the shard's commit point
      critic-0/step-30/ …json …complete
      rollout-0/step-30.json …complete

Heavy shards are directories written by DCP collectively on every rank; the
``.json`` / ``.complete`` sidecars are written by the replica master only,
after the collective returns. A shard without its marker is torn and is
invisible to :func:`resolve_resume_step`, so a crash mid-save costs one
interval rather than the run.

Nothing here imports torch: the launcher and the topology validation both
import this module before any GPU process exists.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Any, Callable, Iterable, Sequence

from loguru import logger

from meshy.config import GlobalCheckpointConfig

#: Set by ``scripts/launch.py --resume``. Its presence is what allows a run to
#: adopt a non-empty state directory; without it a reused ``--runtime-dir`` is
#: an error rather than a silent resume (see
#: ``meshy.service.topology._validate_checkpoint``).
RESUME_ENV = "XRL_RESUME"
#: The resolved resume step, published by the Ignitor before it builds any
#: Service and inherited by every engine subprocess. Resolving it once is not
#: just an optimisation: CPU Services are built *after* the GPU ones are ready,
#: so a second directory scan could in principle see a step the run has since
#: written and give two roles different answers.
RESUME_STEP_ENV = "XRL_RESUME_STEP"

_STEP_DIR_RE = re.compile(r"^step-(\d+)$")
_COMPLETE_RE = re.compile(r"^step-(\d+)\.complete$")


def resume_requested() -> bool:
    """Whether this process was launched with ``--resume``."""
    return os.environ.get(RESUME_ENV, "").strip().lower() not in ("", "0", "false", "no")


class StateStore:
    """Filesystem access to one run's global-checkpoint shards."""

    def __init__(self, state_dir: str) -> None:
        self.root = os.path.abspath(state_dir)

    # ── paths ────────────────────────────────────────────────────────────
    def service_dir(self, service: str) -> str:
        return os.path.join(self.root, service)

    def shard_dir(self, service: str, step: int) -> str:
        """Directory a heavy (DCP) shard is written into."""
        return os.path.join(self.service_dir(service), f"step-{int(step)}")

    def meta_path(self, service: str, step: int) -> str:
        return os.path.join(self.service_dir(service), f"step-{int(step)}.json")

    def _complete_path(self, service: str, step: int) -> str:
        return os.path.join(self.service_dir(service), f"step-{int(step)}.complete")

    # ── writing ──────────────────────────────────────────────────────────
    def write_meta(self, service: str, step: int, payload: dict[str, Any]) -> str:
        """Write the shard's sidecar metadata atomically. Returns its path."""
        path = self.meta_path(service, step)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        document = {
            "service": service,
            "global_step": int(step),
            "created_at": time.time(),
            "payload": payload,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return path

    def mark_complete(self, service: str, step: int) -> None:
        """Commit the shard. Call last, after the heavy write and the meta."""
        path = self._complete_path(service, step)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(str(int(step)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    # ── reading ──────────────────────────────────────────────────────────
    def read_meta(self, service: str, step: int) -> dict[str, Any] | None:
        try:
            with open(self.meta_path(service, step), encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
            return None

    def payload(self, service: str, step: int) -> dict[str, Any]:
        document = self.read_meta(service, step) or {}
        return dict(document.get("payload") or {})

    def complete_steps(self, service: str) -> list[int]:
        """Committed steps for one service, ascending."""
        try:
            entries = os.listdir(self.service_dir(service))
        except (FileNotFoundError, NotADirectoryError):
            return []
        steps = [int(m.group(1)) for m in map(_COMPLETE_RE.match, entries) if m]
        return sorted(steps)

    def is_complete(self, service: str, step: int) -> bool:
        return os.path.exists(self._complete_path(service, step))

    def known_services(self) -> list[str]:
        try:
            return sorted(
                name
                for name in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, name))
            )
        except (FileNotFoundError, NotADirectoryError):
            return []

    def has_any_shard(self) -> bool:
        return any(self.complete_steps(service) for service in self.known_services())

    # ── pruning ──────────────────────────────────────────────────────────
    def prune(self, service: str, keep_latest_k: int) -> list[int]:
        """Drop all but the newest ``keep_latest_k`` committed steps.

        Each service prunes only its own directory, right after committing --
        no coordination needed. The retained sets converge because every
        service uses the same interval and the same ``k``; transiently a peer
        that has not caught up simply keeps the intersection one step older.
        """
        if keep_latest_k <= 0:
            return []
        steps = self.complete_steps(service)
        stale = steps[:-keep_latest_k]
        for step in stale:
            self.drop(service, step)
        return stale

    def drop_newer_than(self, step: int) -> dict[str, list[int]]:
        """Discard shards past the resume point. Call once, before restarting.

        A crash usually leaves one role a window ahead of the others -- the
        critic runs before the trainer, so its shard for the unfinished window
        is already committed. Those shards are provably unusable (if all
        participants had reached that step it would *be* the resume point) and
        the resumed run will rewrite the same step numbers, so keeping them
        only burns a rotation slot and makes the directory lie about how far
        the run got.
        """
        dropped: dict[str, list[int]] = {}
        for service in self.known_services():
            stale = [s for s in self.complete_steps(service) if s > step]
            for candidate in stale:
                self.drop(service, candidate)
            if stale:
                dropped[service] = stale
        return dropped

    def drop(self, service: str, step: int) -> None:
        """Remove one shard: marker first, so a partial delete is never adopted."""
        for path in (
            self._complete_path(service, step),
            self.meta_path(service, step),
        ):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        shutil.rmtree(self.shard_dir(service, step), ignore_errors=True)


def resolve_resume_step(
    store: StateStore,
    participants_at: Callable[[int], Sequence[str]],
    *,
    candidates: Iterable[int] | None = None,
) -> int:
    """Newest step at which every participant committed a shard (0 = none).

    ``participants_at`` must be a pure function of the recipe so that every
    process agrees. It takes a step because participation is step-dependent:
    during the critic's cold start the actor trainer is idle and writes no
    shard, and a resume inside the cold start must still be possible.
    """
    if candidates is None:
        seen: set[int] = set()
        for service in store.known_services():
            seen.update(store.complete_steps(service))
        candidates = seen
    for step in sorted(candidates, reverse=True):
        if step <= 0:
            continue
        required = participants_at(step)
        if required and all(store.is_complete(name, step) for name in required):
            return step
    return 0


def run_resume_step(topology: Any, runtime: Any) -> int:
    """The step every Service should restore to, or 0 for a fresh run.

    Answered from :data:`RESUME_STEP_ENV` when the Ignitor has already
    published it (:meth:`meshy.service.ignite.Ignitor._check_resume`), so every
    role in the run agrees by construction rather than by re-deriving. Falls
    back to a fresh scan for callers outside a full ignition, such as tests.
    """
    if not resume_requested():
        return 0
    cached = os.environ.get(RESUME_STEP_ENV)
    if cached:
        return int(cached)
    return resolve_resume_step(
        StateStore(runtime.state_dir), topology.checkpoint_participants
    )


def publish_resume_step(step: int) -> None:
    """Freeze the run's resume step for every Service and child process."""
    os.environ[RESUME_STEP_ENV] = str(int(step))


def log_resume(service: str, step: int, detail: str = "") -> None:
    """Uniform resume log line, so every role's restore is greppable."""
    if step <= 0:
        logger.info("Checkpoint: {} starting fresh (no resume point)", service)
    else:
        logger.info(
            "Checkpoint: {} resuming at global step {}{}",
            service,
            step,
            f" ({detail})" if detail else "",
        )


__all__ = [
    "GlobalCheckpointConfig",
    "RESUME_ENV",
    "RESUME_STEP_ENV",
    "StateStore",
    "log_resume",
    "publish_resume_step",
    "resolve_resume_step",
    "resume_requested",
    "run_resume_step",
]
