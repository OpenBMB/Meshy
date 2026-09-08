"""Titan training Service (card-level SPMD): Ignition-layer wiring only.

Each card of the training replica spawns one engine subprocess through the
:class:`~meshy.service.spmd.SpmdService` template. This module owns nothing but
the wiring: :meth:`from_info` resolves the topology into constructor fields,
:meth:`build_engine` constructs the training-configured
:class:`~meshy.engine.titan.TitanEngine` (``publish_weights=True``) inside the
child process, and :meth:`build_worker` hands it to a
:class:`~meshy.worker.titan.TitanWorker` together with the colocation
manager. Compute, weight sync and the TQ data plane live in those two layers.

Colocate topologies must be members of a ``COLOCATIONS`` ring: GPU hand-off is
scheduled exclusively by the ring :class:`~meshy.service.colocation.ColocationManager`
(the legacy trainer-driven HTTP arbitration path was removed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from meshy.service.base import GPU
from meshy.service.runtime import RuntimeDir
from meshy.service.spmd import SpmdService, require_ring

if TYPE_CHECKING:
    from meshy.engine.titan import TitanEngine
    from meshy.service.topology import ServiceInfo, Topology
    from meshy.worker.titan import TitanWorker


class TitanTrainingService(SpmdService):
    def __init__(
        self,
        *,
        name: str,
        my_gpu: GPU,
        replica_gpus: list[GPU],
        endpoint_port: int,
        dist_port: int,
        bind_host: str = "0.0.0.0",
        is_colocate: bool = False,
        colocate_inference_names: list[str] | None = None,
        colocate_inference_by_rank: dict[int, str] | None = None,
        inference_targets: list[dict[str, Any]] | None = None,
        weight_sync_mode: str = "auto",
        hf_save_interval: int = 0,
        model_path: str,
        trainer_config: Any,
        trainer_params: Any = None,
        batch_size: int,
        timer_enabled: bool = True,
        stream_minibatch: bool = False,
        tq_endpoints_file: str | None = None,
        partition_id: str | None = None,
        tq_fields: list[str] | None = None,
        tq_poll_interval: float = 0.5,
        runtime: RuntimeDir,
        role: str = "training",
    ) -> None:
        super().__init__(
            name=name,
            role=role,
            my_gpu=my_gpu,
            replica_gpus=replica_gpus,
            endpoint_port=endpoint_port,
            dist_port=dist_port,
            bind_host=bind_host,
            is_colocate=is_colocate,
            runtime=runtime,
        )
        self.colocate_inference_names = colocate_inference_names or []
        self.colocate_inference_by_rank = colocate_inference_by_rank or {}
        self.inference_targets = inference_targets or []
        self.weight_sync_mode = weight_sync_mode
        self.hf_save_interval = hf_save_interval
        self.model_path = model_path
        self.trainer_config = trainer_config
        self.trainer_params = trainer_params
        self.batch_size = batch_size
        self.timer_enabled = timer_enabled
        self.stream_minibatch = stream_minibatch
        self.tq_endpoints_file = tq_endpoints_file
        self.partition_id = partition_id
        self.tq_fields = list(tq_fields or [])
        self.tq_poll_interval = tq_poll_interval

    @classmethod
    def from_info(
        cls,
        info: "ServiceInfo",
        my_gpu: GPU | None,
        topology: "Topology",
        runtime: RuntimeDir,
    ) -> "TitanTrainingService":
        from meshy.config import TrainingServiceConfig
        from meshy.transferqueue.client import resolve_endpoints_file

        config = info.config
        assert isinstance(config, TrainingServiceConfig), (
            f"training service {info.name!r} needs a TrainingServiceConfig, "
            f"got {type(config).__name__}"
        )
        require_ring(info)
        names, by_rank, _endpoints = topology.colocated_inference_for(info)
        return cls(
            name=info.name,
            my_gpu=my_gpu,
            replica_gpus=info.replica_gpus,
            endpoint_port=info.endpoint_port,
            dist_port=info.dist_port,
            is_colocate=info.is_colocate,
            colocate_inference_names=names,
            colocate_inference_by_rank=by_rank,
            inference_targets=topology.inference_targets(),
            weight_sync_mode=str(config.weight_sync_mode),
            hf_save_interval=int(config.hf_save_interval),
            model_path=config.model_path,
            trainer_config=config.trainer_config,
            trainer_params=config.trainer_params,
            batch_size=int(config.batch_size),
            timer_enabled=bool(config.timer_enabled),
            stream_minibatch=bool(config.stream_minibatch),
            tq_endpoints_file=config.tq_endpoints_file or resolve_endpoints_file(runtime.root),
            partition_id=config.partition_id,
            tq_fields=list(config.tq_fields),
            tq_poll_interval=float(config.tq_poll_interval),
            runtime=runtime,
        )

    # ── SpmdService hooks (run in the CHILD process) ─────────────────────
    def build_engine(self) -> "TitanEngine":
        from meshy.engine.spmd import resolve_visible_device
        from meshy.engine.titan import TitanEngine, params_dict

        ring = self.colocation_config
        return TitanEngine(
            rank=self.rank_in_replica,
            world_size=len(self.replica_gpus),
            local_device_id=resolve_visible_device(self.my_gpu.local_rank),
            master_addr=self.master_gpu.host,
            master_port=self.dist_port,
            runtime_root=self.runtime.root,
            name=self.name,
            model_path=self.model_path,
            trainer_config=self.trainer_config,
            trainer_params=params_dict(self.trainer_params),
            timer_enabled=self.timer_enabled,
            batch_size=self.batch_size,
            stream_minibatch=self.stream_minibatch,
            is_colocate=self.is_colocate,
            publish_weights=True,
            inference_targets=self.inference_targets,
            managed_inference_names=self.colocate_inference_names,
            weight_sync_mode="disk" if self.weight_sync_mode == "auto" else self.weight_sync_mode,
            hf_save_interval=self.hf_save_interval,
        )

    def build_worker(self, engine: "TitanEngine", colocation) -> "TitanWorker":
        from meshy.worker.titan import TitanWorker

        # Under the streamed mini-batch schedule the worker fetches one
        # ``mini_batch_size * dp_size`` chunk at a time and only syncs once a
        # full ``batch_size`` has been consumed.
        fetch_batch_size = (
            int(engine.train_chunk_size) if engine.stream_minibatch else self.batch_size
        )
        return TitanWorker(
            engine=engine,
            endpoints_ref=self.tq_endpoints_file,
            partition_id=self.partition_id,
            tq_fields=self.tq_fields,
            batch_size=self.batch_size,
            fetch_batch_size=fetch_batch_size,
            poll_interval=self.tq_poll_interval,
            colocation=colocation,
        )
