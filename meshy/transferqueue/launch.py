"""Spawn and reap the standalone TransferQueue controller / storage processes.

Meshy runs TQ out-of-process so that rollout and trainer can both
reach the same data plane over ZMQ. The launcher starts these processes once
(node 0 only) before ``torchrun``. Endpoint discovery goes through the run's
bootstrap store by default (a ``store://<key>`` ref, see
:mod:`meshy.service.bootstrap`): every component publishes its own key -- no
read-modify-write races, no shared filesystem -- and once all components are up
the cluster consolidates the roster into the single ref key that clients wait
on. A plain file path selects the legacy shared-JSON-file discovery instead
(used with an externally managed TQ via ``XRL_TQ_ENDPOINTS``).

Two non-obvious constraints shape this module:

**File mode serializes startup on purpose.** ``tq_rayless.launcher`` publishes
to the file with an unlocked read-modify-write (read the JSON, add one entry,
atomically replace). Concurrent publishers therefore lose updates -- and a
client that starts with a short storage roster does not fail, it silently
mis-routes, because placement is ``global_idx % num_units``. So in file mode we
start the controller, wait for it to appear in the file, then start each
storage unit and wait for it, one at a time. Store mode has per-component keys,
so storage units start concurrently.

**``TQ_PRE_ALLOC_SAMPLE_NUM`` belongs to the controller.** It is read in
``controller.py`` when a partition is created (``_create_partition``) and nowhere
else, so it must be in the *controller subprocess'* environment; exporting it in
a client process has no effect. It must be at least the per-round sample count,
otherwise consumers cannot tell "not produced yet" from "does not exist" while a
round is still filling.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import IO

from meshy.service.base import die_with_parent

_CLI = "tq_rayless.cli"


class TransferQueueCluster:
    """Owns the controller / storage subprocesses for one run."""

    def __init__(
        self,
        endpoints_ref: str,
        *,
        num_storage_units: int = 1,
        storage_unit_size: int = 100_000,
        pre_alloc_sample_num: int,
        log_dir: str | None = None,
        python: str | None = None,
    ) -> None:
        from meshy.transferqueue.client import is_store_ref, store_key

        self.store_mode = is_store_ref(endpoints_ref)
        if self.store_mode:
            self.endpoints_ref = endpoints_ref
            self.store_key = store_key(endpoints_ref)
            if log_dir is None:
                raise ValueError("log_dir is required with a store:// endpoints ref")
        else:
            self.endpoints_ref = os.path.abspath(endpoints_ref)
        self.num_storage_units = num_storage_units
        self.storage_unit_size = storage_unit_size
        self.pre_alloc_sample_num = pre_alloc_sample_num
        self.log_dir = log_dir or os.path.dirname(self.endpoints_ref)
        self.python = python or sys.executable
        self.procs: list[subprocess.Popen] = []
        self._logs: list[IO[str]] = []

    # Old name, kept for callers that treat the ref as a file path.
    @property
    def endpoints_file(self) -> str:
        return self.endpoints_ref

    # -- process plumbing ---------------------------------------------------
    def _spawn(self, name: str, argv: list[str], env: dict[str, str]) -> subprocess.Popen:
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"{name}.log")
        log = open(log_path, "w")
        self._logs.append(log)
        proc = subprocess.Popen(
            [self.python, "-m", _CLI, *argv],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=die_with_parent,
        )
        self.procs.append(proc)
        return proc

    def _base_env(self) -> dict[str, str]:
        return os.environ.copy()

    def _wait_published(self, predicate, what: str, proc: subprocess.Popen, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{what} exited with code {proc.returncode} before publishing; "
                    f"see {self.log_dir}"
                )
            try:
                with open(self.endpoints_file) as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}
            if predicate(data):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{what} did not publish within {timeout}s; see {self.log_dir}")
            time.sleep(0.05)

    # -- lifecycle ----------------------------------------------------------
    def start(self, timeout: float = 120.0) -> "TransferQueueCluster":
        if self.store_mode:
            return self._start_store(timeout)
        return self._start_file(timeout)

    def _start_store(self, timeout: float) -> "TransferQueueCluster":
        from meshy.service import bootstrap

        # Resolving the store here also exports XRL_BOOTSTRAP_ADDR, so both the
        # TQ subprocesses (below, via argv) and any worker process spawned
        # later reach the same store.
        store = bootstrap.get_store()
        addr = os.environ["XRL_BOOTSTRAP_ADDR"]

        # Best-effort cleanup of keys from a previous cluster under the same
        # ref (same-process reuse, e.g. pytest): a leftover key would make the
        # waits below pass against dead endpoints.
        for key in self._component_keys() + [self.store_key]:
            try:
                store.delete_key(key)
            except Exception:
                pass

        env = self._base_env()
        env["TQ_PRE_ALLOC_SAMPLE_NUM"] = str(self.pre_alloc_sample_num)
        # Without polling mode an under-filled get_meta blocks on a fixed sleep
        # and then raises TimeoutError, so a consumer polling for a partially
        # produced batch would crash instead of being told "not yet".
        controller = self._spawn(
            "tq-controller",
            ["controller", "--publish-store", addr, "--publish-key", self.store_key,
             "--polling-mode"],
            env,
        )
        self._wait_store_keys([f"{self.store_key}/controller"], "tq-controller", [controller], timeout)

        # Per-component keys cannot race each other, so storage units start
        # concurrently (unlike file mode's serialized read-modify-write).
        storage_env = self._base_env()
        storages = [
            self._spawn(
                f"tq-storage-{rank}",
                ["storage", "--publish-store", addr, "--publish-key", self.store_key,
                 "--rank", str(rank), "--size", str(self.storage_unit_size)],
                storage_env,
            )
            for rank in range(self.num_storage_units)
        ]
        self._wait_store_keys(
            [f"{self.store_key}/storage/{r}" for r in range(self.num_storage_units)],
            "tq-storage", storages, timeout,
        )

        # Consolidate the roster under the ref key itself: clients wait on this
        # single key, so they can never observe a partial storage roster.
        bootstrap.set_json(self.store_key, {
            "controller": bootstrap.get_json(f"{self.store_key}/controller"),
            "storage": {
                str(r): bootstrap.get_json(f"{self.store_key}/storage/{r}")
                for r in range(self.num_storage_units)
            },
        })
        print(
            f"[tq] ready: controller + {self.num_storage_units} storage unit(s), "
            f"pre_alloc={self.pre_alloc_sample_num} endpoints={self.endpoints_ref}",
            flush=True,
        )
        return self

    def _component_keys(self) -> list[str]:
        return [f"{self.store_key}/controller"] + [
            f"{self.store_key}/storage/{r}" for r in range(self.num_storage_units)
        ]

    def _wait_store_keys(
        self, keys: list[str], what: str, procs: list[subprocess.Popen], timeout: float
    ) -> None:
        from meshy.service import bootstrap

        def liveness() -> None:
            for proc in procs:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{what} exited with code {proc.returncode} before publishing; "
                        f"see {self.log_dir}"
                    )

        try:
            bootstrap.wait_keys(keys, timeout=timeout, interval=0.05, liveness=liveness)
        except TimeoutError as exc:
            raise TimeoutError(f"{what} did not publish within {timeout}s; see {self.log_dir}") from exc

    def _start_file(self, timeout: float) -> "TransferQueueCluster":
        # A stale file from a previous run would make _wait_published return
        # immediately against dead endpoints.
        if os.path.exists(self.endpoints_ref):
            os.remove(self.endpoints_ref)
        os.makedirs(os.path.dirname(self.endpoints_ref), exist_ok=True)

        env = self._base_env()
        env["TQ_PRE_ALLOC_SAMPLE_NUM"] = str(self.pre_alloc_sample_num)
        controller = self._spawn(
            "tq-controller",
            ["controller", "--endpoints-file", self.endpoints_ref, "--polling-mode"],
            env,
        )
        self._wait_published(lambda d: bool(d.get("controller")), "tq-controller", controller, timeout)

        storage_env = self._base_env()
        for rank in range(self.num_storage_units):
            proc = self._spawn(
                f"tq-storage-{rank}",
                [
                    "storage",
                    "--endpoints-file",
                    self.endpoints_ref,
                    "--rank",
                    str(rank),
                    "--size",
                    str(self.storage_unit_size),
                ],
                storage_env,
            )
            self._wait_published(
                lambda d, r=rank: str(r) in (d.get("storage") or {}),
                f"tq-storage-{rank}",
                proc,
                timeout,
            )

        print(
            f"[tq] ready: controller + {self.num_storage_units} storage unit(s), "
            f"pre_alloc={self.pre_alloc_sample_num} endpoints={self.endpoints_ref}",
            flush=True,
        )
        return self

    def stop(self, grace: float = 10.0) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + grace
        for proc in self.procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
        for log in self._logs:
            try:
                log.close()
            except Exception:
                pass
        self.procs.clear()
        self._logs.clear()

    def __enter__(self) -> "TransferQueueCluster":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
