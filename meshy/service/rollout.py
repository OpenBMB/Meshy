"""Rollout Service: spawns the rollout driver process (0 GPU).

Launched once by the global rank-0 ignitor. The Service is pure Ignition
layer: it resolves the topology wiring in :meth:`from_info`, spawns one driver
subprocess in :meth:`ignite`, and inside that child builds the
:class:`~meshy.engine.sglang.SGLangEngine` and hands it to the
:class:`~meshy.worker.rollout.RolloutWorker` -- all rollout policy (pacing,
strategies, TQ writes) lives in the Worker.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from typing import TYPE_CHECKING

from loguru import logger

from meshy.service.base import (
    GPU,
    Service,
    die_with_parent,
    raise_open_file_limit,
    redirect_output,
)
from meshy.service.runtime import RuntimeDir

if TYPE_CHECKING:
    from meshy.engine.sglang import SGLangEngine
    from meshy.service.topology import ServiceInfo, Topology
    from meshy.worker.rollout import RolloutWorker


def _build_rollout_engine(kwargs: dict):
    from meshy.engine.sglang import SGLangEngine
    endpoints = list(kwargs["inference_endpoints"])
    sampling_params = kwargs.get("sampling_params", {})
    return SGLangEngine(endpoints, sampling_params)


def _build_rollout_worker(
    kwargs: dict, engine, colocation=None, *, runtime_root: str | None = None
):
    from meshy.worker.rollout import RolloutWorker

    trajectory_log = kwargs.get("trajectory_log")
    if not trajectory_log and runtime_root:
        trajectory_log = RuntimeDir(runtime_root).trajectory_path()
    return RolloutWorker(
        engine=engine,
        endpoints_ref=kwargs["tq_endpoints_file"],
        model_path=kwargs["model_path"],
        dataset=kwargs["dataset"],
        dataset_kwargs=kwargs.get("dataset_kwargs", {}),
        partition_id=kwargs.get("partition_id", "data.train"),
        group_size=int(kwargs.get("group_size", 1)),
        train_batch_size=int(kwargs["train_batch_size"]),
        sampling_params=kwargs.get("sampling_params", {}),
        reward=kwargs.get("reward") or "meshy.worker.rollout:grpo_advantage",
        advantage=kwargs.get("advantage"),
        advantage_kwargs=kwargs.get("advantage_kwargs", {}),
        filter_zero_std_groups=bool(kwargs.get("filter_zero_std_groups", False)),
        num_epochs=int(kwargs.get("num_epochs", 1)),
        pacing_window=kwargs.get("pacing_window", 1),
        max_running_requests=int(kwargs.get("async_max_running_request", -1) or -1),
        poll_interval=float(kwargs.get("poll_interval", 2.0)),
        trajectory_log=trajectory_log,
        verbose_trajectory_log=bool(kwargs.get("verbose_trajectory_log", False)),
        colocation=colocation,
    )


async def _amain(kwargs: dict, runtime: RuntimeDir) -> None:
    engine = _build_rollout_engine(kwargs)
    worker = _build_rollout_worker(kwargs, engine, runtime_root=runtime.root)
    await worker.run_async()


class RolloutService(Service):
    def __init__(self, *, name: str, kwargs: dict, runtime: RuntimeDir) -> None:
        super().__init__(name=name, role="rollout")
        self.kwargs = kwargs
        self.runtime = runtime
        self.engine: "SGLangEngine | None" = None
        self.worker: "RolloutWorker | None" = None

    @classmethod
    def from_info(
        cls,
        info: "ServiceInfo",
        my_gpu: GPU | None,
        topology: "Topology",
        runtime: RuntimeDir,
    ) -> "RolloutService":
        from dataclasses import asdict

        from meshy.config import RolloutServiceConfig, TrainingServiceConfig
        from meshy.transferqueue.client import resolve_endpoints_file

        config = info.config
        assert isinstance(config, RolloutServiceConfig), (
            f"rollout service {info.name!r} needs a RolloutServiceConfig, "
            f"got {type(config).__name__}"
        )
        # The driver currently consumes a plain dict; the typed config is
        # flattened whole so nothing a recipe sets can be dropped, and the
        # topology-derived wiring is injected alongside.
        kwargs = asdict(config)
        kwargs["inference_endpoints"] = topology.inference_endpoints()
        kwargs["tq_endpoints_file"] = config.tq_endpoints_file or resolve_endpoints_file(
            runtime.root
        )
        # The trainer's batch size == samples one gen gate releases; the pacer
        # needs it to convert the gate stream into a sample budget.
        trainers = topology.training_services()
        assert trainers, "rollout requires a training service in the topology"
        trainer_config = trainers[0].config
        assert isinstance(trainer_config, TrainingServiceConfig)
        kwargs["train_batch_size"] = int(trainer_config.batch_size)
        return cls(name=info.name, kwargs=kwargs, runtime=runtime)

    def ignite(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=self._run_runtime)
        p.start()
        self.processes.append(p)
        self.is_master = True

    def wait_for_ready(self) -> None:
        # No HTTP server; readiness == process spawned. Publish a marker file.
        self.runtime.mark_ready(self.name)

    def _run_runtime(self) -> None:
        """Child-process entry: build the Engine, hand it to the Worker, run."""
        log_path = self.runtime.process_log_path(self.name, 0)
        redirect_output(log_path)
        logger.info("{} {} rank 0 started; output redirected to {}", self.role, self.name, log_path)
        from meshy.worker.rollout import RolloutWorker

        die_with_parent()
        soft = raise_open_file_limit()
        logger.info("Rollout: RLIMIT_NOFILE soft limit set to {}", soft)
        self.engine = _build_rollout_engine(self.kwargs)
        self.engine.model_path = self.kwargs.get("model_path")
        colocation = self.build_colocation_manager(
            on_acquire=self.engine.on_colocate_acquire,
            on_release=self.engine.on_colocate_release,
        )
        colocation.start()
        self.worker = _build_rollout_worker(
            self.kwargs, self.engine, colocation, runtime_root=self.runtime.root
        )
        asyncio.run(self.worker.run_async())
