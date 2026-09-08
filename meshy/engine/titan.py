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
):
    """Build Meshy's TitanTrainer metrics adapter."""
    from meshy.backend.titan import TitanTrainer, build_forge_config
    from meshy.utils.model import resolve_model_path

    local_path = resolve_model_path(model_path)
    forge_config = build_forge_config(trainer_config, hf_model_path=local_path)
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
        hf_save_interval: int = 0,
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
        self.hf_save_interval = int(hf_save_interval or 0)

        self.trainer = None
        self.train_chunk_size: int | None = None
        self.step_index = 0
        self.weight_version = 0
        self.last_weights_path: str | None = None

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
        )
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
            self.save_checkpoint(weights_path)
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

    def save_checkpoint(self, path: str) -> None:
        """Save the current model as an HF checkpoint on all ranks."""
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
            "weight_version": self.weight_version,
            "ready": self.ready,
        }

    def close(self) -> None:
        if self.trainer is not None and hasattr(self.trainer, "close"):
            self.trainer.close()
        self.trainer = None
        super().close()
