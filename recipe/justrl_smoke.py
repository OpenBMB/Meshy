"""Bounded 16x-smaller GPU smoke test for :mod:`recipe.justrl`.

This keeps the production 8-card colocated topology and two real GRPO
train/sync cycles, while limiting the rollout to two 16-prompt batches (256
samples total, versus 2048 per batch in ``recipe.justrl``).
"""

from __future__ import annotations

from meshy.config import (
    InferenceServiceConfig,
    RolloutServiceConfig,
    SamplingParams,
    TrainerConfig,
    TrainerParamsConfig,
    TrainingServiceConfig,
)
from meshy.dataset.math import MATH
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor

MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
NUM_INFERENCE_ENGINES = 8
ROLLOUT_BATCH = 16
GROUP_SIZE = 8
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # 128, 16x below justrl
NUM_EPOCHS = 1


class SmokeMATH(MATH):
    """Expose exactly two prompt batches so the smoke run covers two train/sync cycles."""

    def __init__(self, batch_size: int, seed: int | None = None, **kwargs):
        super().__init__(batch_size=batch_size, seed=seed, **kwargs)
        self._batches_left = 2

    def next_batch(self, builder):
        if self._batches_left <= 0:
            return []
        self._batches_left -= 1
        return super().next_batch(builder)


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name="qwen2_5_math",
        model_flavor="R1-Distill-1.5B",
        seq_len=512,
        lr=1e-6,
        weight_decay=0.1,
        max_norm=1.0,
        warmup_steps=0,
        steps=1,
        dtype="bfloat16",
        compile_model=False,
        dp_shard_degree=-1,
        dp_replicate_degree=8,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=False,
        dump_folder="./outputs/justrl_smoke",
    )


def _trainer_params() -> TrainerParamsConfig:
    return TrainerParamsConfig(
        # One local micro-batch keeps the two-window smoke run short while
        # retaining the production global batch and colocation transitions.
        mini_batch_size=8,
        micro_batch_size=1,
        ppo_clip_eps_low=0.2,
        ppo_clip_eps_high=0.28,
        old_logprobs_source="train",
    )


def _inference_group() -> ServiceGroup:
    return ServiceGroup(
        id="actor_infer",
        n_replicas=NUM_INFERENCE_ENGINES,
        n_gpus_per_replica=1,
        config=InferenceServiceConfig(
            model_path=MODEL_PATH,
            server_args={
                "model_path": MODEL_PATH,
                "tp_size": 1,
                "enable_memory_saver": True,
                "mem_fraction_static": 0.6,
            },
        ),
    )


def _training_group() -> ServiceGroup:
    return ServiceGroup(
        id="actor_train",
        n_replicas=1,
        n_gpus_per_replica=NUM_INFERENCE_ENGINES,
        colocate_with="actor_infer",
        wait_until=["actor_infer"],
        config=TrainingServiceConfig(
            model_path=MODEL_PATH,
            trainer_config=_trainer_config(),
            trainer_params=_trainer_params(),
            batch_size=BATCH_SIZE,
            timer_enabled=True,
        ),
    )


def _rollout_group() -> ServiceGroup:
    return ServiceGroup(
        id="rollout",
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_train", "actor_infer"],
        config=RolloutServiceConfig(
            model_path=MODEL_PATH,
            dataset="recipe.justrl_smoke:SmokeMATH",
            dataset_kwargs={"batch_size": ROLLOUT_BATCH, "seed": 42},
            reward="meshy.dataset.math:MATH.reward",
            sampling_params=SamplingParams(max_new_tokens=128).as_dict(),
            group_size=GROUP_SIZE,
            poll_interval=1.0,
            pacing_window=1,
            num_epochs=NUM_EPOCHS,
        ),
    )


def build_service_groups() -> list[ServiceGroup]:
    return [_inference_group(), _training_group(), _rollout_group()]


SERVICE_GROUPS = build_service_groups()
COLOCATIONS = [
    ColocationRing(
        group_id="actor_card",
        ring=(
            ("actor_infer", SchedulingMode.FALLBACK),
            ("actor_train", SchedulingMode.ON_DEMAND),
        ),
    )
]


def main() -> None:
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
