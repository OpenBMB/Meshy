"""SPMD engines of the OPD role: the Student trainer and the Titan Teacher.

Both are :class:`~meshy.engine.titan.TitanEngine` subclasses.  The Student
engine differs only in the trainer class it constructs; the Teacher engine
adds one SPMD command (``score``) next to the inherited residency commands.
"""

from __future__ import annotations

from typing import Any

import torch

from meshy.engine.titan import TitanEngine


def _build_student_trainer(
    model_path: str, trainer_config: Any, trainer_params: dict[str, Any] | None,
    timer_enabled: bool, runtime_root: str | None, name: str,
):
    # Same recipe as ``meshy.engine.titan.build_titan_trainer`` with the role's
    # trainer class; duplicated because that helper hardcodes ``TitanTrainer``.
    from meshy.backend.titan import build_forge_config
    from meshy.service.runtime import RuntimeDir
    from meshy.utils.model import resolve_model_path

    from meshy.backend.titan.opd import StudentTopKTrainer

    forge_config = build_forge_config(trainer_config, hf_model_path=resolve_model_path(model_path))
    runtime = RuntimeDir(runtime_root) if runtime_root else None
    return StudentTopKTrainer(
        forge_config,
        timer_enabled=timer_enabled,
        tensorboard_log_dir=(runtime.tensorboard_path(name) if runtime else None),
        **(trainer_params or {}),
    )


class StudentTopKEngine(TitanEngine):
    """TitanEngine whose trainer is :class:`StudentTopKTrainer`."""

    def setup(self) -> None:
        self.trainer = _build_student_trainer(
            self.model_path, self.trainer_config, self.trainer_params,
            self.timer_enabled, runtime_root=self.runtime.root, name=self.name,
        )
        mesh = self.trainer.parallel_dims.get_optional_mesh("batch")
        dp_size = mesh.size() if mesh is not None else 1
        self.train_chunk_size = self.trainer.mini_batch_size * dp_size
        if self.batch_size is not None and self.batch_size % dp_size != 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be divisible by DP size ({dp_size})")
        if self.is_colocate:
            self.trainer.offload_to_cpu()
            torch.cuda.empty_cache()


class OPDTeacherEngine(TitanEngine):
    """A frozen Titan model that returns Top-K ids / log-probs for rollout rows.

    ``score`` runs through the replica command loop like ``step``: the master
    broadcasts the rows, every DP rank scores its token-balanced share using
    the trainer's own planner and micro-batch layout, and the results are
    gathered back to the master.
    """

    def __init__(self, *, top_k: int, **kwargs: Any) -> None:
        super().__init__(publish_weights=False, inference_targets=[], **kwargs)
        self.top_k = int(top_k)

    def score(self, samples: list[Any]) -> list[dict[str, torch.Tensor]] | None:
        return self.submit_command("score", samples=samples)

    def execute(self, payload: dict[str, Any], samples: list[Any] | None) -> Any:
        if payload.get("action") == "score":
            return self._score_impl(samples)
        return super().execute(payload, samples)

    def _score_impl(self, samples: list[Any] | None) -> list[dict[str, torch.Tensor]] | None:
        from meshy.backend.titan.parallel import dp_rank_and_size

        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("OPDTeacherEngine.score() called before init()")
        batch = self._broadcast({"samples": samples} if self.is_master else None)["samples"]
        dp_rank, dp_size = dp_rank_and_size(trainer.parallel_dims)
        results = score_samples(trainer, batch, self.top_k, dp_rank=dp_rank, dp_size=dp_size)

        if self.world_size > 1:
            import torch.distributed as dist

            gathered: list[Any] = [None] * self.world_size
            dist.all_gather_object(gathered, results, group=self.group_gloo)
            if not self.is_master:
                return None
            for part in gathered:
                results.update(part)
        missing = [i for i in range(len(batch)) if i not in results]
        if missing:
            raise RuntimeError(f"Teacher scoring left {len(missing)} rows unscored: {missing[:8]}")
        return [
            {"teacher_topk_ids": results[i][0], "teacher_topk_logprobs": results[i][1]}
            for i in range(len(batch))
        ]


def score_samples(
    trainer: Any, batch: list[Any], top_k: int, *, dp_rank: int = 0, dp_size: int = 1
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Score this DP rank's share of ``batch`` -> ``{global index: (ids, logprobs)}``.

    Uses the trainer's own planner and micro-batch layout, so the Teacher
    forward sees exactly the token windows the Student will train on.  Every
    DP rank must call this with the same ``batch`` (the plan is deterministic).
    """
    from meshy.backend.titan.opd import split_teacher_topk
    from meshy.backend.titan.topk_loss import teacher_topk

    if getattr(trainer.parallel_dims, "cp_enabled", False):
        raise NotImplementedError("OPD Teacher scoring does not support context parallelism")
    plan = trainer.plan_batch(batch, dp_size)
    local = plan.local_samples(batch, dp_rank)
    global_idx = plan.local_indices[dp_rank]
    results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad(), trainer.train_context():
        for mini in plan.per_rank[dp_rank]:
            for micro in mini.micros:
                mb = trainer._build_micro(local, micro, need_rollout_lp=False)
                logits = trainer.model_parts[0](
                    mb.input_ids, positions=mb.positions, attention_masks=mb.attention_masks
                )
                ids, lps = teacher_topk(logits, top_k, chunk=trainer._LOGPROB_CHUNK)
                del logits
                ids_per = split_teacher_topk(ids, micro, layout=trainer.batch_layout)
                lps_per = split_teacher_topk(lps, micro, layout=trainer.batch_layout)
                for j, local_pos in enumerate(micro.sample_idx):
                    results[global_idx[local_pos]] = (
                        ids_per[j].to("cpu", torch.int64), lps_per[j].to("cpu", torch.float32)
                    )
                del mb, ids, lps
    return results


__all__ = ["OPDTeacherEngine", "StudentTopKEngine", "score_samples"]
