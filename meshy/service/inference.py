"""SGLang server Service for Meshy.

A Service owns card placement and the SGLang server subprocess.  Generation and
server management HTTP calls live in :class:`meshy.engine.sglang.SGLangEngine`.
Only the replica's node-local leader launches a server; the replica master waits
for all endpoints to become ready and publishes the runtime marker.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable

from loguru import logger

from meshy.engine.sglang import SGLangEngine
from meshy.service.base import GPU, Service
from meshy.service.runtime import RuntimeDir

if TYPE_CHECKING:
    from meshy.service.topology import ServiceInfo, Topology

_TORCHRUN_ENV_KEYS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
    "LOCAL_WORLD_SIZE", "GROUP_WORLD_SIZE", "ROLE_WORLD_SIZE", "ROLE_NAME",
    "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_ERROR_FILE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
)


def _augment_path(env: dict[str, str]) -> None:
    extra = [os.path.dirname(sys.executable)]
    cuda_home = (
        env.get("XRL_CUDA_HOME")
        or env.get("CUDA_HOME")
        or next((p for p in ("/usr/local/cuda", "/usr/local/cuda-12.9") if os.path.isdir(p)), None)
    )
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        extra.append(os.path.join(cuda_home, "bin"))
    existing = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([*extra, existing]) if existing else os.pathsep.join(extra)


def _server_args_to_cli(server_args: dict) -> list[str]:
    argv: list[str] = []
    for key, value in server_args.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif value is None:
            continue
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        else:
            argv.extend((flag, str(value)))
    return argv


class _Subprocess:
    """Small ``Popen`` adapter matching the Service process lifecycle API."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def join(self, timeout: float | None = None) -> None:
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self) -> None:
        if not self.is_alive():
            return
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            self._process.terminate()


class SGLangService(Service):
    """Placement and lifecycle wrapper around one SGLang replica."""

    def __init__(
        self,
        *,
        name: str,
        my_gpu: GPU,
        replica_gpus: list[GPU],
        server_args: dict,
        endpoint_port: int,
        dist_port: int,
        bind_host: str = "0.0.0.0",
        is_colocate: bool = False,
        group_endpoints: list[str] | None = None,
        model_path: str | None = None,
        runtime: RuntimeDir,
    ) -> None:
        super().__init__(name=name, role="inference")
        if not replica_gpus:
            raise ValueError("SGLangService requires at least one replica GPU")
        self.my_gpu = my_gpu
        self.replica_gpus = list(replica_gpus)
        self.server_args = dict(server_args)
        self.endpoint_port = int(endpoint_port)
        self.dist_port = int(dist_port)
        self.bind_host = bind_host
        self.is_colocate = bool(is_colocate)
        self.runtime = runtime
        self.master_gpu = self.replica_gpus[0]
        self.host = self.master_gpu.host
        self.is_master = my_gpu.global_rank == self.master_gpu.global_rank
        self.rank_in_service = [gpu.global_rank for gpu in self.replica_gpus].index(
            my_gpu.global_rank
        )
        self.model_path = model_path or self.server_args.get("model_path")
        # This engine owns management calls for this service's one SGLang
        # process.  Rollout constructs its own client over the full replica
        # endpoint set; sharing that set here would make every colocated
        # service release every peer's memory concurrently.
        endpoints = list(group_endpoints or [])
        self.endpoint = f"http://{self.host}:{self.endpoint_port}"
        # Each replica manages its own server during readiness. The
        # colocation leader additionally needs a full-replica client because a
        # token hand-off restores/releases the whole inference pool at once.
        self.engine = SGLangEngine([self.endpoint])
        self.engine.model_path = self.model_path
        self._group_endpoints = endpoints
        self.colocation_engine = None
        self._log_file = None
        self.colocation_manager = None

    @classmethod
    def from_info(
        cls,
        info: "ServiceInfo",
        my_gpu: GPU | None,
        topology: "Topology",
        runtime: RuntimeDir,
    ) -> "SGLangService":
        from meshy.config import InferenceServiceConfig

        if my_gpu is None:
            raise ValueError(f"SGLang service {info.name!r} requires a GPU")
        config = info.config
        if not isinstance(config, InferenceServiceConfig):
            raise TypeError(
                f"inference service {info.name!r} needs InferenceServiceConfig, "
                f"got {type(config).__name__}"
            )
        return cls(
            name=info.name,
            my_gpu=my_gpu,
            replica_gpus=info.replica_gpus,
            server_args=dict(config.server_args),
            endpoint_port=info.endpoint_port,
            dist_port=info.dist_port,
            is_colocate=info.is_colocate,
            group_endpoints=[member.endpoint for member in topology.group(info.group_id)],
            model_path=config.model_path,
            runtime=runtime,
        )

    def _child_env(self, devices: Iterable[int]) -> dict[str, str]:
        env = os.environ.copy()
        original = env.get("CUDA_VISIBLE_DEVICES")
        mapped = [original.split(",")[device] for device in devices] if original else [str(device) for device in devices]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(mapped)
        env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        for key in _TORCHRUN_ENV_KEYS:
            env.pop(key, None)
        _augment_path(env)
        return env

    def ignite(self) -> None:
        by_node: dict[int, list[GPU]] = defaultdict(list)
        for gpu in self.replica_gpus:
            by_node[gpu.node_rank].append(gpu)
        node_gpus = sorted(by_node[self.my_gpu.node_rank], key=lambda gpu: gpu.global_rank)
        node_leader = min(node_gpus, key=lambda gpu: gpu.global_rank)
        if self.my_gpu.global_rank != node_leader.global_rank:
            logger.info("SGLang {} passive on rank {}", self.name, self.my_gpu.global_rank)
            return

        args = dict(self.server_args)
        if self.model_path is not None:
            args.setdefault("model_path", self.model_path)
        args.update({"host": self.bind_host, "port": self.endpoint_port, "tp_size": len(self.replica_gpus)})
        if len(by_node) > 1:
            args.update({
                "nnodes": len(by_node),
                "node_rank": sorted(by_node).index(self.my_gpu.node_rank),
                "dist_init_addr": f"{self.master_gpu.host}:{self.dist_port}",
            })
        devices = [gpu.local_rank for gpu in node_gpus]
        command = [sys.executable, "-m", "sglang.launch_server", *_server_args_to_cli(args)]
        log_path = self.runtime.process_log_path(self.name, self.rank_in_service)
        self._log_file = open(log_path, "ab", buffering=0)
        logger.info("Igniting SGLang {} (devices {}, endpoint {}, log {}): {}", self.name, devices, self.endpoint, log_path, " ".join(command))
        process = subprocess.Popen(
            command,
            env=self._child_env(devices),
            start_new_session=True,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        self.processes.append(_Subprocess(process))

    def _assert_processes_alive(self) -> None:
        for process in self.processes:
            if not process.is_alive():
                raise RuntimeError(f"SGLang {self.name} process exited before ready")

    def wait_for_ready(self) -> None:
        if not self.is_master:
            return
        logger.info("Waiting for SGLang {} at {}", self.name, self.endpoint)
        self.engine.wait_ready(liveness=self._assert_processes_alive)
        if self.is_colocate:
            logger.info("Colocate inference {} releasing memory", self.name)
            self.engine.release_for_colocate()
            if self.is_colocation_leader:
                if self.colocation_engine is None:
                    self.colocation_engine = SGLangEngine(self._group_endpoints or [self.endpoint])
                    self.colocation_engine.model_path = self.model_path
                manager_engine = self.colocation_engine
                self.colocation_manager = self.build_colocation_manager(
                    on_acquire=manager_engine.on_colocate_acquire,
                    on_release=manager_engine.on_colocate_release,
                )
                self.colocation_manager.start()
        self.runtime.mark_ready(self.name)
        logger.info("SGLang {} ready; marker published", self.name)

    def terminate(self) -> None:
        if self.colocation_manager is not None:
            self.colocation_manager.stop()
            self.colocation_manager = None
        for process in self.processes:
            process.terminate()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        try:
            asyncio.run(self.engine.close())
            if self.colocation_engine is not None:
                asyncio.run(self.colocation_engine.close())
        except RuntimeError:
            # A caller already owning an event loop can close the async client.
            pass
        super().terminate()
