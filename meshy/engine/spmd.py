"""Shared SPMD engine skeleton for GPU compute backends.

Every GPU Engine has the same skeleton: one spawned subprocess per card, an
independent per-replica process group (NCCL for compute + a separate gloo
group for control), a rank-0 command queue whose payloads are broadcast to all
ranks so collective work stays symmetric, and a rank-0 command dispatcher.
:class:`SpmdEngine` owns exactly that skeleton and nothing role- or
backend-specific. The containing Service builds the Engine in the engine
subprocess and hands it to its Worker; the Engine has no dependency on either.
Backends subclass it and implement two hooks:

* :meth:`SpmdEngine.setup` -- build the model/trainer (all ranks; called after
  the process group exists, before the engine reports ready);
* :meth:`SpmdEngine.execute` -- dispatch one broadcast command (all ranks).
"""

from __future__ import annotations

import os
import threading
from datetime import timedelta
from queue import Queue
from typing import Any

from loguru import logger

from meshy.service.runtime import RuntimeDir

# torchrun/torchelastic rendezvous vars scrubbed before the engine builds its own
# per-replica process group (see :meth:`SpmdEngine._setup_env`). Leaving
# ``TORCHELASTIC_USE_AGENT_STORE`` set makes ``init_process_group`` attach to the
# ignitor's agent store and hang.
_TORCHELASTIC_ENV_KEYS = (
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_ERROR_FILE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "GROUP_RANK",
    "ROLE_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "ROLE_NAME",
)

# Long-context rollout generation can leave non-master ranks waiting for the
# next rank-0 command for longer than PyTorch's 30-minute default. Keep the
# timeout configurable for shorter jobs while giving long-running SPMD jobs a
# generous default. This applies to both the compute process group and the
# dedicated Gloo control group below.
_DEFAULT_PROCESS_GROUP_TIMEOUT_S = 8 * 60 * 60


def _process_group_timeout() -> timedelta:
    value = float(
        os.environ.get(
            "XRL_SPMD_TIMEOUT_S", str(_DEFAULT_PROCESS_GROUP_TIMEOUT_S)
        )
    )
    if value <= 0:
        raise ValueError("XRL_SPMD_TIMEOUT_S must be positive")
    return timedelta(seconds=value)


def resolve_visible_device(device_id: int) -> str:
    """Map a topology-local device ordinal through ``CUDA_VISIBLE_DEVICES``."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(device_id)
    return visible.split(",")[device_id]


class _PendingCommand:
    def __init__(
        self, action: str, samples: list[Any] | None = None, sync: bool = True
    ) -> None:
        self.action = action
        self.samples = samples
        # For a ``train`` command: whether this step closes a ``batch_size``
        # window and must therefore dump a checkpoint, sync weights to
        # inference and bump the version. Always ``True`` outside the streamed
        # mini-batch schedule.
        self.sync = sync
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None


class SpmdEngine:
    """Engine-subprocess skeleton: environment, process group and command loop.

    Rank 0 is the only rank that originates work through :meth:`submit_command`.
    The command payload is broadcast on a dedicated gloo group and every rank
    executes it symmetrically so collective calls inside :meth:`execute` line
    up. HTTP and process management belong to the containing Service.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_device_id: str,
        master_addr: str,
        master_port: int,
        runtime_root: str,
        name: str,
        bind_host: str = "",
        endpoint_port: int = 0,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.local_device_id = local_device_id
        self.master_addr = master_addr
        self.master_port = master_port
        self.bind_host = bind_host
        self.endpoint_port = endpoint_port
        self.name = name
        self.runtime = RuntimeDir(runtime_root)

        self.command_queue: "Queue[_PendingCommand]" = Queue()
        self.group_gloo = None
        self.ready = False

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    # ── backend hooks ────────────────────────────────────────────────────
    def setup(self) -> None:  # pragma: no cover - interface
        """Build the backend's model/trainer. Runs on every rank."""
        raise NotImplementedError

    def execute(self, payload: dict, samples: list[Any] | None) -> Any:
        """Execute one broadcast command. Runs on every rank."""
        raise NotImplementedError  # pragma: no cover - interface

    def health_info(self) -> dict[str, Any]:
        """Extra fields merged into the ``/health`` payload."""
        return {}

    def on_colocate_acquire(self, grant: Any) -> None:
        """Restore backend state after this Engine receives the GPU token."""
        raise NotImplementedError

    def on_colocate_release(self, target: str) -> None:
        """Release backend state before publishing a GPU token."""
        raise NotImplementedError

    # ── lifecycle ────────────────────────────────────────────────────────
    def init(self) -> None:
        """Initialize distributed state and the backend-specific compute."""
        self._setup_env()
        self._init_process_group()
        self.setup()
        self.ready = True

    def start_command_loop(self) -> threading.Thread:
        """Start rank 0's command dispatcher after initialization."""
        if not self.is_master:
            raise RuntimeError("only rank 0 can start the command loop thread")
        thread = threading.Thread(
            target=self._command_loop,
            name=f"{self.name}-commands",
            daemon=True,
        )
        thread.start()
        return thread

    def run_command_loop(self) -> None:
        """Run the blocking command loop used by non-master ranks."""
        self._command_loop()

    def _setup_env(self) -> None:
        # Scrub the outer torchrun/torchelastic rendezvous so this engine's own
        # (independent, per-replica) process-group init doesn't try to attach to
        # the ignitor's elastic-agent store -- with a mismatched port/topology
        # that attach hangs forever. We then set a clean, self-contained env.
        for key in _TORCHELASTIC_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.local_device_id
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)

    def _init_process_group(self) -> None:
        import torch
        import torch.distributed as dist

        timeout = _process_group_timeout()
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size,
                timeout=timeout,
            )
        # Separate gloo group for command broadcast / barriers so they don't
        # race an in-flight NCCL collective on the compute stream.
        self.group_gloo = dist.new_group(backend="gloo", timeout=timeout)

    def submit_command(
        self, action: str, samples: list[Any] | None = None, sync: bool = True
    ) -> Any:
        """Queue a command on rank 0 and wait for symmetric execution."""
        if not self.is_master:
            raise RuntimeError("submit_command must be called on the replica master")
        pending = _PendingCommand(action=action, samples=samples, sync=sync)
        self.command_queue.put(pending)
        pending.done.wait()
        if pending.error is not None:
            raise pending.error
        return pending.result

    def _command_loop(self) -> None:
        while True:
            pending: _PendingCommand | None = None
            if self.is_master:
                pending = self.command_queue.get()
                payload = {"action": pending.action, "sync": pending.sync}
            else:
                payload = None
            payload = self._broadcast(payload)
            try:
                result = self.execute(payload, pending.samples if pending else None)
                if pending is not None:
                    pending.result = result
            except BaseException as exc:  # noqa: BLE001 - surfaced to caller
                # Keep the traceback visible even when the rank-0 caller owns a
                # pending command; otherwise TQ only reports a generic fatal.
                logger.exception("{} engine command failed", self.name)
                if pending is not None:
                    pending.error = exc
            finally:
                if pending is not None:
                    pending.done.set()

    def _broadcast(self, payload: dict | None) -> dict:
        if self.world_size == 1:
            assert payload is not None
            return payload
        import torch.distributed as dist

        container = [payload]
        dist.broadcast_object_list(container, src=0, group=self.group_gloo)
        return container[0]

    def close(self) -> None:
        """Release backend resources; subclasses may extend this hook."""
        self.ready = False

    def _barrier(self) -> None:
        if self.world_size > 1:
            import torch.distributed as dist

            dist.barrier(group=self.group_gloo)
