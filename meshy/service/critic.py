"""Ignition-layer wiring of the critic role.

:class:`CriticService` is an :class:`~meshy.service.spmd.SpmdService` around
:class:`~meshy.engine.critic.CriticSpmdEngine` and
:class:`~meshy.worker.critic.CriticWorker` -- the same shape as
:class:`~meshy.service.opd.OPDTeacherService`, because the two roles have the
same job description: an SPMD replica that consumes rollout rows and writes a
column back.

The critic normally gets its own cards (``justrl_ii_recipe.md`` §3 splits nodes
evenly between actor and critic), in which case it is not a ring member and
needs no GPU arbitration. Colocation is supported for small single-node runs;
:func:`require_ring` enforces that a colocated critic actually declares a ring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from meshy.service.base import GPU
from meshy.service.runtime import RuntimeDir
from meshy.service.spmd import SpmdService, require_ring

if TYPE_CHECKING:
    from meshy.service.topology import ServiceInfo, Topology


class CriticService(SpmdService):
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
        score_batch_size: int,
        publish_mode: str,
        micro_rows: int,
        max_tokens_per_micro: int | None,
        train_backbone: bool,
        lr: float,
        warmup_steps: int,
        cold_start_windows: int,
        value_epochs: int,
        gamma: float,
        alpha: float,
        max_norm: float,
        timer_enabled: bool,
        tq_endpoints_file: str,
        partition_id: str,
        tq_poll_interval: float,
        publish_gate_zero: bool = False,
        actor_name: str | None = None,
        actor_model_path: str | None = None,
        checkpoint: Any = None,
        resume_step: int = 0,
    ) -> None:
        super().__init__(
            name=name, role="critic", my_gpu=my_gpu, replica_gpus=replica_gpus,
            endpoint_port=endpoint_port, dist_port=dist_port,
            is_colocate=is_colocate, runtime=runtime,
        )
        self.model_path = model_path
        self.trainer_config = trainer_config
        self.score_batch_size = score_batch_size
        self.publish_mode = publish_mode
        self.micro_rows = micro_rows
        self.max_tokens_per_micro = max_tokens_per_micro
        self.train_backbone = train_backbone
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.cold_start_windows = cold_start_windows
        self.publish_gate_zero = publish_gate_zero
        self.value_epochs = value_epochs
        self.gamma = gamma
        self.alpha = alpha
        self.max_norm = max_norm
        self.timer_enabled = timer_enabled
        self.tq_endpoints_file = tq_endpoints_file
        self.partition_id = partition_id
        self.tq_poll_interval = tq_poll_interval
        self.actor_name = actor_name
        self.actor_model_path = actor_model_path
        self.checkpoint = checkpoint
        self.resume_step = int(resume_step)

    @classmethod
    def from_info(
        cls, info: "ServiceInfo", my_gpu: GPU | None, topology: "Topology", runtime: RuntimeDir
    ):
        from meshy.transferqueue.client import resolve_endpoints_file

        from meshy.config import CriticServiceConfig
        from meshy.service.checkpoint import run_resume_step

        config = info.config
        assert isinstance(config, CriticServiceConfig), (
            f"critic service {info.name!r} needs a CriticServiceConfig, "
            f"got {type(config).__name__}"
        )
        require_ring(info)
        # Only needed when the critic shares cards with the inference server:
        # the card goes back to SGLang after a critic window and the grant has
        # to name the weights it should reload (see ``CriticWorker``).
        trainers = topology.training_services()
        actor = trainers[0] if trainers else None
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
            score_batch_size=int(config.score_batch_size),
            publish_mode=str(config.publish_mode),
            micro_rows=int(config.micro_rows),
            max_tokens_per_micro=(
                int(config.max_tokens_per_micro)
                if config.max_tokens_per_micro
                else None
            ),
            train_backbone=bool(config.train_backbone),
            lr=float(config.lr),
            warmup_steps=int(config.warmup_steps),
            cold_start_windows=int(config.cold_start_windows),
            publish_gate_zero=bool(config.publish_gate_zero),
            value_epochs=int(config.value_epochs),
            gamma=float(config.gamma),
            alpha=float(config.alpha),
            max_norm=float(config.max_norm),
            timer_enabled=bool(config.timer_enabled),
            tq_endpoints_file=config.tq_endpoints_file or resolve_endpoints_file(runtime.root),
            partition_id=config.partition_id,
            tq_poll_interval=float(config.tq_poll_interval),
            actor_name=actor.name if actor else None,
            actor_model_path=getattr(actor.config, "model_path", None) if actor else None,
            checkpoint=config.checkpoint,
            resume_step=run_resume_step(topology, runtime),
        )

    def _actor_weights(self, version: int) -> str | None:
        """Checkpoint the inference server must hold after a critic window.

        ``version`` is the newest actor weight version the scored rows were
        generated against. Version 0 is the base model the SGLang servers
        booted from -- during the critic's cold start the actor never steps, so
        this is the only path the whole phase ever returns.
        """
        if version <= 0:
            return self.actor_model_path or self.model_path
        if self.actor_name is None:
            return None
        return self.runtime.checkpoint_path(self.actor_name, version)

    # ── SpmdService hooks (run in the CHILD process) ─────────────────────
    def build_engine(self):
        from meshy.engine.spmd import resolve_visible_device

        from meshy.engine.critic import CriticSpmdEngine

        return CriticSpmdEngine(
            rank=self.rank_in_replica,
            world_size=len(self.replica_gpus),
            local_device_id=resolve_visible_device(self.my_gpu.local_rank),
            master_addr=self.master_gpu.host,
            master_port=self.dist_port,
            runtime_root=self.runtime.root,
            name=self.name,
            model_path=self.model_path,
            trainer_config=self.trainer_config,
            train_backbone=self.train_backbone,
            lr=self.lr,
            warmup_steps=self.warmup_steps,
            value_epochs=self.value_epochs,
            gamma=self.gamma,
            alpha=self.alpha,
            max_norm=self.max_norm,
            micro_rows=self.micro_rows,
            max_tokens_per_micro=self.max_tokens_per_micro,
            publish_mode=self.publish_mode,
            is_colocate=self.is_colocate,
            checkpoint=self.checkpoint,
            resume_step=self.resume_step,
        )

    def build_worker(self, engine, colocation):
        from meshy.worker.critic import CriticWorker

        return CriticWorker(
            engine=engine,
            endpoints_ref=self.tq_endpoints_file,
            partition_id=self.partition_id,
            score_batch_size=self.score_batch_size,
            cold_start_windows=self.cold_start_windows,
            publish_mode=self.publish_mode,
            publish_gate_zero=self.publish_gate_zero,
            poll_interval=self.tq_poll_interval,
            colocation=colocation,
            actor_weights=self._actor_weights if self.is_colocate else None,
            resume_step=self.resume_step,
        )


__all__ = ["CriticService"]
