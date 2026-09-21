"""TorchTitan-backed compute engine for Meshy.

The engine owns model construction, distributed execution, GPU residency and
checkpoint creation. It deliberately knows nothing about TransferQueue or the
role-specific Worker that drives it. A replica master calls :meth:`step`; the
SPMD command loop broadcasts the operation to every rank.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from loguru import logger

from meshy.engine.spmd import SpmdEngine


@dataclass
class StepResult:
    """Result of a completed training window (meaningful on rank 0)."""

    metrics: dict[str, Any]
    step: int
    weight_version: int
    weights_path: str | None
    synced: bool


def params_dict(trainer_params: Any) -> dict[str, Any]:
    """Normalize a dataclass, mapping, or ``None`` into trainer kwargs."""
    from dataclasses import asdict, is_dataclass

    if trainer_params is None:
        return {}
    if is_dataclass(trainer_params):
        return asdict(trainer_params)
    return dict(trainer_params)


def build_titan_trainer(
    model_path: str,
    trainer_config: Any,
    trainer_params: dict[str, Any] | None,
    timer_enabled: bool,
    runtime_root: str | None = None,
    name: str = "titan",
    *,
    state_dir: str | None = None,
    resume_step: int = 0,
):
    """Build Meshy's TitanTrainer metrics adapter.

    ``state_dir`` points the DCP checkpointer at this service's global-
    checkpoint directory; when it already contains ``step-{resume_step}`` the
    trainer restores model + optimizer + LR schedule + its own step counter
    instead of doing the initial HF load.
    """
    from meshy.backend.titan import TitanTrainer, build_forge_config
    from meshy.utils.model import resolve_model_path

    local_path = resolve_model_path(model_path)
    forge_config = build_forge_config(
        trainer_config,
        hf_model_path=local_path,
        state_dir=state_dir,
        resume_step=resume_step,
    )
    from meshy.service.runtime import RuntimeDir
    runtime = RuntimeDir(runtime_root) if runtime_root else None
    return TitanTrainer(
        forge_config,
        timer_enabled=timer_enabled,
        tensorboard_log_dir=(runtime.tensorboard_path(name) if runtime else None),
        **(trainer_params or {}),
    )


def _publish_weights_to_inference(
    targets: list[dict[str, Any]],
    weights_path: str,
    version: int,
    name: str,
    *,
    managed_names: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Load a completed HF checkpoint into inference servers outside the ring.

    In a colocation ring the next GPU grant carries ``weights_path`` and the
    SGLang acquire callback reloads it after the inference process regains its
    card.  Those endpoints must not be touched here: they have deliberately
    released their weight buffers.  A topology may still contain additional,
    disaggregated inference replicas; those are updated over HTTP immediately.
    """
    targets = [target for target in targets if target.get("name") not in managed_names]
    if not targets:
        return
    import httpx
    import time

    started = time.monotonic()

    def sync_one(target: dict[str, Any]) -> None:
        endpoint = str(target["endpoint"]).rstrip("/")
        response = httpx.post(
            f"{endpoint}/update_weights_from_disk",
            json={"model_path": weights_path},
            timeout=1800.0,
        )
        response.raise_for_status()

    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = [pool.submit(sync_one, target) for target in targets]
        for future in futures:
            future.result()
    logger.info(
        "Titan {} synced weights v{} to {} inference engine(s) in {:.1f}s",
        name,
        version,
        len(targets),
        time.monotonic() - started,
    )


class TitanEngine(SpmdEngine):
    """SPMD training engine with a small role-independent public API."""

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
        model_path: str,
        trainer_config: Any,
        trainer_params: dict[str, Any] | None = None,
        timer_enabled: bool = True,
        batch_size: int | None = None,
        stream_minibatch: bool = False,
        is_colocate: bool = False,
        publish_weights: bool = True,
        inference_targets: list[dict[str, Any]] | None = None,
        managed_inference_names: list[str] | None = None,
        weight_sync_mode: str = "disk",
        weights_keep_latest_k: int = 0,
        checkpoint: Any = None,
        gate_step_offset: int = 0,
        resume_step: int = 0,
        bind_host: str = "127.0.0.1",
        endpoint_port: int = 0,
    ) -> None:
        super().__init__(
            rank=rank,
            world_size=world_size,
            local_device_id=local_device_id,
            master_addr=master_addr,
            master_port=master_port,
            bind_host=bind_host,
            endpoint_port=endpoint_port,
            runtime_root=runtime_root,
            name=name,
        )
        if weight_sync_mode not in ("disk", "auto"):
            raise ValueError(
                "Meshy TitanEngine currently supports only disk weight sync; "
                f"got {weight_sync_mode!r}"
            )
        self.model_path = model_path
        self.trainer_config = trainer_config
        self.trainer_params = trainer_params or {}
        self.timer_enabled = bool(timer_enabled)
        self.batch_size = batch_size
        self.stream_minibatch = bool(stream_minibatch)
        self.is_colocate = bool(is_colocate)
        self.publish_weights = bool(publish_weights)
        self.inference_targets = list(inference_targets or [])
        self.managed_inference_names = set(managed_inference_names or [])
        self.weight_sync_mode = weight_sync_mode
        self.weights_keep_latest_k = int(weights_keep_latest_k or 0)

        from meshy.config import GlobalCheckpointConfig

        self.checkpoint = GlobalCheckpointConfig.coerce(checkpoint)
        #: Windows the critic consumes before this trainer's first step; the
        #: offset between this engine's ``step_index`` and the run's global
        #: window index (see :meth:`global_step`).
        self.gate_step_offset = int(gate_step_offset)
        #: Global window index this engine restores to; 0 for a fresh run.
        self.resume_step = int(resume_step)

        self.trainer = None
        self.train_chunk_size: int | None = None
        self.step_index = 0
        self.weight_version = 0
        self.last_weights_path: str | None = None
        self._store = None

    @property
    def global_step(self) -> int:
        """This engine's step on the run's shared clock (the gate stream).

        The critic's cold-start windows come before the trainer's first step,
        so the trainer's own counter lags the global one by exactly that
        offset -- the same offset ``TitanWorker`` adds to its gate numbers.
        """
        return self.step_index + self.gate_step_offset

    def init(self) -> None:
        """Initialize SPMD state and construct TitanTrainer."""
        super().init()

    def setup(self) -> None:
        self.trainer = build_titan_trainer(
            self.model_path,
            self.trainer_config,
            self.trainer_params,
            self.timer_enabled,
            runtime_root=self.runtime.root,
            name=self.name,
            state_dir=self._state_dir(),
            resume_step=self.resume_step,
        )
        self._restore_counters()
        mesh = self.trainer.parallel_dims.get_optional_mesh("batch")
        dp_size = mesh.size() if mesh is not None else 1
        self.train_chunk_size = self.trainer.mini_batch_size * dp_size
        if self.batch_size is not None and self.batch_size % dp_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by DP size ({dp_size})"
            )
        if self.is_colocate:
            # SGLang owns the card at genesis.  The legacy trainer performed
            # this initial conversion before the first token hand-off; without
            # it Titan remains resident after setup and the genesis acquire
            # can OOM while restoring the inference weights.
            self.trainer.offload_to_cpu()
            import torch

            torch.cuda.empty_cache()

    def step(
        self,
        samples: list[Any] | None,
        *,
        sync: bool = True,
    ) -> StepResult | None:
        """Run one training operation through the replica command loop.

        Only rank 0 invokes this method. The full batch stays on rank 0 until
        :meth:`execute` scatters it, keeping Worker and TQ concerns out of the
        Engine.
        """
        return self.submit_command("step", samples=samples, sync=bool(sync))

    def execute(self, payload: dict[str, Any], samples: list[Any] | None) -> StepResult | None:
        action = payload.get("action")
        if action == "colocate_acquire":
            self.restore()
            return None
        if action == "colocate_release":
            self.offload()
            return None
        if action != "step":
            raise ValueError(f"unknown TitanEngine action: {action!r}")
        return self._step_impl(samples, sync=bool(payload.get("sync", True)))

    def _step_impl(self, samples: list[Any] | None, *, sync: bool) -> StepResult | None:
        if self.trainer is None:
            raise RuntimeError("TitanEngine.step() called before init()")

        from meshy.backend.titan.parallel import split_batch_to_local

        if self.world_size == 1:
            local_batch, local_plan = samples, None
        else:
            # Every rank plans the same broadcast batch; the plan decides the
            # token-balanced DP subset and the mini/micro schedule.
            local_batch, local_plan = split_batch_to_local(
                full_batch=samples if self.is_master else None,
                parallel_dims=self.trainer.parallel_dims,
                group=self.group_gloo,
                src=0,
                planner=self.trainer.plan_batch,
            )

        # Rollout statistics are computed over the *global* batch on the
        # master (the only rank that logs); other ranks only see their slice.
        metrics = self.trainer.train_step(
            local_batch,
            plan=local_plan,
            step_schedule=sync,
            rollout_samples=samples if self.is_master else None,
        )
        self._barrier()

        if not sync:
            return (
                StepResult(
                    metrics=dict(metrics) if isinstance(metrics, dict) else {"value": metrics},
                    step=self.step_index,
                    weight_version=self.weight_version,
                    weights_path=self.last_weights_path,
                    synced=False,
                )
                if self.is_master
                else None
            )

        new_step = self.step_index + 1
        weights_path = self._weights_path(new_step) if self.publish_weights else None
        if weights_path is not None:
            self.export_hf_weights(weights_path)
            self.last_weights_path = weights_path
            # Colocated inference receives this path through the next GPU grant
            # (the trainer must release the card before inference can resume).
            # Disaggregated inference can load it immediately over its HTTP API.
            if self.is_master:
                _publish_weights_to_inference(
                    self.inference_targets,
                    weights_path,
                    new_step,
                    self.name,
                    managed_names=self.managed_inference_names,
                )
        self.step_index = new_step
        self.weight_version = new_step

        # Still holding the GPU token here, and every rank is inside the same
        # broadcast command -- the only point in the window where a collective
        # DCP save is both safe and cheap (the model is resident, so nothing
        # has to be paged back from CPU).
        if self.checkpoint.due(self.global_step):
            self._save_global_checkpoint(self.global_step)
        self._prune_weights(new_step)

        result = StepResult(
            metrics=dict(metrics) if isinstance(metrics, dict) else {"value": metrics},
            step=self.step_index,
            weight_version=self.weight_version,
            weights_path=self.last_weights_path,
            synced=True,
        )
        logger.info(
            "TitanEngine {} completed step {} (weight v{})",
            self.name,
            self.step_index,
            self.weight_version,
        )
        return result if self.is_master else None

    def _weights_path(self, version: int) -> str:
        return self.runtime.checkpoint_path(self.name, version)

    # ── global checkpoint ────────────────────────────────────────────────
    def _state_dir(self) -> str | None:
        """Where this service's DCP shards live, or ``None`` when disabled."""
        import os

        if not self.checkpoint.enabled:
            return None
        return os.path.join(self.runtime.state_dir, self.name)

    @property
    def store(self):
        """Lazily-built :class:`~meshy.service.checkpoint.StateStore`."""
        if self._store is None:
            self._store = self.runtime.state_store()
        return self._store

    def _restore_counters(self) -> None:
        """Re-seed the engine's own counters from the restored shard.

        ``TitanTrainer.step`` comes back through DCP, but ``step_index`` /
        ``weight_version`` / ``last_weights_path`` are the engine's and are not
        in any state dict. They are read from the shard's sidecar rather than
        derived from ``trainer.step``: the two coincide today but diverge under
        ``stream_minibatch``, and silently exporting ``v1`` over a previous
        run's ``v1`` is not a failure anything downstream would notice.
        """
        from meshy.service.checkpoint import log_resume

        if self.resume_step <= 0:
            return
        if not self.store.is_complete(self.name, self.resume_step):
            # Resuming inside the critic's cold start: the actor has not taken
            # a step, so it is not a participant at this step and has nothing
            # on disk. The DCP folder is absent too, so the trainer has
            # already done its initial HF load and the zeroed counters below
            # are correct. Anything later is a genuinely missing shard.
            if self.gate_step_offset and self.resume_step <= self.gate_step_offset:
                if self.is_master:
                    log_resume(
                        self.name,
                        self.resume_step,
                        "still in the critic's cold start; the actor has not "
                        "stepped, so it starts from the base model",
                    )
                return
            raise FileNotFoundError(
                f"resuming at global step {self.resume_step} but "
                f"{self.store.meta_path(self.name, self.resume_step)} is missing; "
                "the shard is incomplete"
            )
        payload = self.store.payload(self.name, self.resume_step)
        self.step_index = int(payload["step_index"])
        self.weight_version = int(payload["weight_version"])
        self.last_weights_path = payload.get("weights_path")
        trainer_step = int(getattr(self.trainer, "step", 0))
        if trainer_step != int(payload["trainer_step"]):
            raise RuntimeError(
                f"{self.name}: DCP restored trainer.step={trainer_step} but the "
                f"shard's metadata records {payload['trainer_step']}. The "
                f"checkpoint directory {self.store.shard_dir(self.name, self.resume_step)} "
                "and its sidecar disagree; they are not from the same run."
            )
        if self.is_master:
            log_resume(
                self.name,
                self.resume_step,
                f"step_index={self.step_index} trainer.step={trainer_step} "
                f"weights={self.last_weights_path}",
            )

    def _save_global_checkpoint(self, global_step: int) -> None:
        """Persist this trainer's shard. Collective: every rank must call."""
        if self.trainer is None:
            raise RuntimeError("TitanEngine is not initialized")
        self.trainer.save_checkpoint(global_step)
        self._barrier()
        if not self.is_master:
            return
        self.store.write_meta(
            self.name,
            global_step,
            {
                "step_index": self.step_index,
                "weight_version": self.weight_version,
                "weights_path": self.last_weights_path,
                "trainer_step": int(getattr(self.trainer, "step", 0)),
            },
        )
        self.store.mark_complete(self.name, global_step)
        self.store.prune(self.name, self.checkpoint.keep_latest_k)
        logger.info(
            "Titan {} checkpointed global step {} -> {}",
            self.name,
            global_step,
            self.store.shard_dir(self.name, global_step),
        )

    def _protected_weight_versions(self) -> set[int]:
        """Exported versions a retained global checkpoint still points at.

        A resumed run hands its inference servers the weights the shard names
        (``SGLangService`` genesis, ``CriticWorker._actor_weights_for``), so
        pruning one of those would make the checkpoint unusable even though
        the DCP shard itself survived.
        """
        protected: set[int] = set()
        for step in self.store.complete_steps(self.name):
            version = self.store.payload(self.name, step).get("weight_version")
            if version is not None:
                protected.add(int(version))
        return protected

    def _prune_weights(self, current: int) -> None:
        """Rotate the exported HF weight directories (rank 0 only).

        One full model is written per step purely so inference can reload it;
        nothing reads an old version once the next one lands. Left alone, a
        few hundred steps of a 2.25B model is terabytes -- long runs have died
        of exactly this.
        """
        import os
        import re
        import shutil

        if self.weights_keep_latest_k <= 0 or not self.is_master:
            return
        directory = self.runtime.weights_dir(self.name)
        try:
            entries = os.listdir(directory)
        except (FileNotFoundError, NotADirectoryError):
            return
        versions = sorted(
            int(m.group(1)) for m in (re.match(r"^v(\d+)$", e) for e in entries) if m
        )
        keep = set(versions[-self.weights_keep_latest_k :])
        keep.update(self._protected_weight_versions())
        keep.add(int(current))
        dropped = [v for v in versions if v not in keep]
        for version in dropped:
            shutil.rmtree(self._weights_path(version), ignore_errors=True)
        if dropped:
            logger.info(
                "Titan {} pruned exported weights {} (kept {})",
                self.name,
                dropped,
                sorted(keep & set(versions)),
            )

    def export_hf_weights(self, path: str) -> None:
        """Export the current model as an HF checkpoint for inference to load.

        This is the weight-sync path, not a resumable checkpoint: it carries
        no optimizer, LR schedule or step. See :meth:`_save_global_checkpoint`
        for the one a resume reads. All ranks must call together.
        """
        if self.trainer is None:
            raise RuntimeError("TitanEngine is not initialized")
        self.trainer.save_hf_checkpoint(path)

    def offload(self) -> None:
        """Move Titan state off GPU and release cached allocations."""
        if self.trainer is None:
            raise RuntimeError("TitanEngine is not initialized")
        import torch

        self.trainer.offload_to_cpu()
        torch.cuda.empty_cache()

    def on_colocate_release(self, target: str) -> None:
        del target
        if self.world_size > 1 and self.is_master:
            self.submit_command("colocate_release")
        else:
            self.offload()

    def on_colocate_acquire(self, grant: Any) -> None:
        del grant
        if self.world_size > 1 and self.is_master:
            self.submit_command("colocate_acquire")
        else:
            self.restore()

    def restore(self) -> None:
        """Restore Titan state to its configured GPU."""
        if self.trainer is None:
            raise RuntimeError("TitanEngine is not initialized")
        self.trainer.restore_to_gpu()

    def health_info(self) -> dict[str, Any]:
        return {
            "step": self.step_index,
            "global_step": self.global_step,
            "weight_version": self.weight_version,
            "resumed_from": self.resume_step,
            "ready": self.ready,
        }

    def close(self) -> None:
        if self.trainer is not None and hasattr(self.trainer, "close"):
            self.trainer.close()
        self.trainer = None
        super().close()
