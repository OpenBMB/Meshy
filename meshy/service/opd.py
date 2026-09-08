"""Ignition-layer wiring of the OPD role: two Services, no new lifecycle code.

:class:`OPDTrainingService` is the stock Titan training Service building the
Student engine; :class:`OPDTeacherService` is an
:class:`~meshy.service.spmd.SpmdService` around the Teacher engine and Worker.
The rollout and inference Services of the recipe are the ordinary ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from meshy.service.base import GPU
from meshy.service.runtime import RuntimeDir
from meshy.service.spmd import SpmdService, require_ring
from meshy.service.training import TitanTrainingService

if TYPE_CHECKING:
    from meshy.service.topology import ServiceInfo, Topology


class OPDTrainingService(TitanTrainingService):
    distill_params: dict[str, Any] = {}

    @classmethod
    def from_info(cls, info: "ServiceInfo", my_gpu: GPU | None, topology: "Topology", runtime: RuntimeDir):
        from meshy.config import OPDTrainingConfig

        config = info.config
        assert isinstance(config, OPDTrainingConfig), (
            f"OPD training service {info.name!r} needs an OPDTrainingConfig, got {type(config).__name__}"
        )
        service = super().from_info(info, my_gpu, topology, runtime)
        service.distill_params = {
            "log_prob_min_clamp": config.log_prob_min_clamp,
            "loss_max_clamp": config.loss_max_clamp,
        }
        return service

    def build_engine(self):
        from meshy.engine.spmd import resolve_visible_device
        from meshy.engine.titan import params_dict

        from meshy.engine.opd import StudentTopKEngine

        # Same wiring as ``TitanTrainingService.build_engine`` (which hardcodes
        # the engine class) with the Student engine and the loss settings.
        return StudentTopKEngine(
            rank=self.rank_in_replica,
            world_size=len(self.replica_gpus),
            local_device_id=resolve_visible_device(self.my_gpu.local_rank),
            master_addr=self.master_gpu.host,
            master_port=self.dist_port,
            runtime_root=self.runtime.root,
            name=self.name,
            model_path=self.model_path,
            trainer_config=self.trainer_config,
            trainer_params={**params_dict(self.trainer_params), **self.distill_params},
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


class OPDTeacherService(SpmdService):
    def __init__(
        self,
        *,
        name: str,
        my_gpu: GPU,
        replica_gpus: list[GPU],
        endpoint_port: int,
        dist_port: int,
        is_colocate: bool,
        runtime: RuntimeDir,
        model_path: str,
        trainer_config: Any,
        top_k: int,
        score_batch_size: int,
        timer_enabled: bool,
        tq_endpoints_file: str,
        partition_id: str,
        tq_poll_interval: float,
        student_name: str | None,
        student_model_path: str | None,
    ) -> None:
        super().__init__(
            name=name, role="teacher", my_gpu=my_gpu, replica_gpus=replica_gpus,
            endpoint_port=endpoint_port, dist_port=dist_port, is_colocate=is_colocate, runtime=runtime,
        )
        self.model_path = model_path
        self.trainer_config = trainer_config
        self.top_k = top_k
        self.score_batch_size = score_batch_size
        self.timer_enabled = timer_enabled
        self.tq_endpoints_file = tq_endpoints_file
        self.partition_id = partition_id
        self.tq_poll_interval = tq_poll_interval
        self.student_name = student_name
        self.student_model_path = student_model_path

    @classmethod
    def from_info(cls, info: "ServiceInfo", my_gpu: GPU | None, topology: "Topology", runtime: RuntimeDir):
        from meshy.transferqueue.client import resolve_endpoints_file

        from meshy.config import OPDTeacherConfig

        config = info.config
        assert isinstance(config, OPDTeacherConfig), (
            f"teacher service {info.name!r} needs an OPDTeacherConfig, got {type(config).__name__}"
        )
        require_ring(info)
        students = (
            topology.group(config.student_group) if config.student_group else topology.training_services()
        )
        student = students[0] if students else None
        return cls(
            name=info.name,
            my_gpu=my_gpu,
            replica_gpus=info.replica_gpus,
            endpoint_port=info.endpoint_port,
            dist_port=info.dist_port,
            is_colocate=info.is_colocate,
            runtime=runtime,
            model_path=config.model_path,
            trainer_config=config.trainer_config,
            top_k=int(config.top_k),
            score_batch_size=int(config.score_batch_size),
            timer_enabled=bool(config.timer_enabled),
            tq_endpoints_file=config.tq_endpoints_file or resolve_endpoints_file(runtime.root),
            partition_id=config.partition_id,
            tq_poll_interval=float(config.tq_poll_interval),
            student_name=student.name if student else None,
            student_model_path=getattr(student.config, "model_path", None) if student else None,
        )

    def _student_weights(self, version: int) -> str | None:
        """Checkpoint the inference server must hold after a Teacher window."""
        if version <= 0:
            return self.student_model_path
        if self.student_name is None:
            return None
        return self.runtime.checkpoint_path(self.student_name, version)

    # ── SpmdService hooks (run in the CHILD process) ─────────────────────
    def build_engine(self):
        from meshy.engine.spmd import resolve_visible_device

        from meshy.engine.opd import OPDTeacherEngine

        return OPDTeacherEngine(
            top_k=self.top_k,
            rank=self.rank_in_replica,
            world_size=len(self.replica_gpus),
            local_device_id=resolve_visible_device(self.my_gpu.local_rank),
            master_addr=self.master_gpu.host,
            master_port=self.dist_port,
            runtime_root=self.runtime.root,
            name=self.name,
            model_path=self.model_path,
            trainer_config=self.trainer_config,
            trainer_params={},
            timer_enabled=self.timer_enabled,
            is_colocate=self.is_colocate,
        )

    def build_worker(self, engine, colocation):
        from meshy.worker.opd import OPDTeacherWorker

        return OPDTeacherWorker(
            engine=engine,
            endpoints_ref=self.tq_endpoints_file,
            partition_id=self.partition_id,
            score_batch_size=self.score_batch_size,
            poll_interval=self.tq_poll_interval,
            colocation=colocation,
            student_weights=self._student_weights,
        )


__all__ = ["OPDTeacherService", "OPDTrainingService"]
