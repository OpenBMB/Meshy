"""GRPO-on-GSM8K recipe for the card-level SPMD Service stack.

Build a ``SERVICE_GROUPS`` list (declarative config) and hand it to the
:class:`~meshy.service.ignite.Ignitor`. Everything is environment-tunable so the
same recipe drives single-GPU smoke tests and multi-GPU runs.

Run with the launcher (recommended)::

    python scripts/launch.py --recipe recipe.grpo_gsm8k

Or directly with a single torchrun (colocate and disaggregate alike)::

    XRL_TOPOLOGY=disaggregate XRL_NGPUS=2 \\
        torchrun --standalone --nproc-per-node 2 -m recipe.grpo_gsm8k

Topologies (``XRL_TOPOLOGY``):
* ``colocate``     -- one colocate pair (inference+training share all cards) +
  AgentLoop.
* ``disaggregate`` -- standalone inference replicas + one standalone training
  replica (separate cards) + AgentLoop.
"""

from __future__ import annotations

import os

from meshy.config import (
    RolloutServiceConfig,
    InferenceServiceConfig,
    TrainerConfig,
    TrainerParamsConfig,
    TrainingServiceConfig,
)
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor

MODEL_PATH = os.environ.get("XRL_MODEL", "Qwen/Qwen3-1.7B")
MODEL_NAME = os.environ.get("XRL_MODEL_NAME", "qwen3")
MODEL_FLAVOR = os.environ.get("XRL_MODEL_FLAVOR", "1.7B")
NGPUS = int(os.environ.get("XRL_NGPUS", "1"))
TOPOLOGY = os.environ.get("XRL_TOPOLOGY", "colocate")
SEQ_LEN = int(os.environ.get("XRL_SEQ_LEN", "2048"))
MAX_NEW_TOKENS = int(os.environ.get("XRL_MAX_NEW_TOKENS", "1024"))

ROLLOUT_BATCH = int(os.environ.get("XRL_ROLLOUT_BATCH", "4"))
GROUP_SIZE = int(os.environ.get("XRL_GROUP_SIZE", "4"))
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # trainer trigger threshold


def _trainer_config() -> TrainerConfig:
    # local_batch_size / global_batch_size are intentionally left at their
    # defaults (see meshy.config.TrainerConfig): they are ForgeEngine hints only
    # -- TitanTrainer does its own micro-batching -- and overriding them risks
    # tripping ForgeEngine's grad-accum assertion. Matches recipe/justrl.
    return TrainerConfig(
        model_name=MODEL_NAME,
        model_flavor=MODEL_FLAVOR,
        seq_len=SEQ_LEN,
        steps=int(os.environ.get("XRL_STEPS", "1000")),
        dtype="bfloat16",
        lr=float(os.environ.get("XRL_LR", "1e-6")),
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=False,
        dump_folder="./outputs/grpo_gsm8k",
        compile_model=False,
    )


def _trainer_params() -> TrainerParamsConfig:
    return TrainerParamsConfig(
        mini_batch_size=1,
        micro_batch_size=1,
        ppo_clip_eps_low=0.2,
        ppo_clip_eps_high=0.2,
        old_logprobs_source="rollout",
    )


def _inference_config(tp_size: int, *, enable_memory_saver: bool) -> InferenceServiceConfig:
    server_args = {"model_path": MODEL_PATH, "tp_size": tp_size}
    if enable_memory_saver:
        # Required so /release_memory_occupation + /resume_memory_occupation work
        # for the colocate GPU hand-off.
        server_args["enable_memory_saver"] = True
    return InferenceServiceConfig(model_path=MODEL_PATH, server_args=server_args)


def _training_config() -> TrainingServiceConfig:
    return TrainingServiceConfig(
        model_path=MODEL_PATH,
        trainer_config=_trainer_config(),
        trainer_params=_trainer_params(),
        batch_size=BATCH_SIZE,
        timer_enabled=True,
    )


def _rollout_group() -> ServiceGroup:
    return ServiceGroup(
        id="rollout",
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_train", "actor_infer"],
        config=RolloutServiceConfig(
            model_path=MODEL_PATH,
            dataset="meshy.dataset.gsm8k:GSM8K",
            dataset_kwargs={"batch_size": ROLLOUT_BATCH, "split": "train", "seed": 42},
            reward="meshy.dataset.gsm8k:GSM8K.reward",
            sampling_params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            group_size=GROUP_SIZE,
            poll_interval=2.0,
            pacing_window=1,
            num_epochs=int(os.environ.get("XRL_EPOCHS", "1")),
        ),
    )


def build_service_groups() -> list[ServiceGroup]:
    if TOPOLOGY == "colocate":
        return [
            ServiceGroup(
                id="actor_infer",
                config=_inference_config(NGPUS, enable_memory_saver=True),
                n_replicas=1,
                n_gpus_per_replica=NGPUS,
            ),
            ServiceGroup(
                id="actor_train",
                config=_training_config(),
                n_replicas=1,
                n_gpus_per_replica=NGPUS,
                colocate_with="actor_infer",
                wait_until=["actor_infer"],
            ),
            _rollout_group(),
        ]

    if TOPOLOGY == "disaggregate":
        if NGPUS < 2:
            raise ValueError("disaggregate topology needs XRL_NGPUS >= 2")
        n_train = int(os.environ.get("XRL_TRAIN_GPUS", str(max(1, NGPUS // 2))))
        n_infer = NGPUS - n_train
        if n_infer < 1:
            raise ValueError("disaggregate needs at least 1 inference card")
        return [
            ServiceGroup(
                id="actor_infer",
                config=_inference_config(1, enable_memory_saver=False),
                n_replicas=n_infer,
                n_gpus_per_replica=1,
            ),
            ServiceGroup(
                id="actor_train",
                config=_training_config(),
                n_replicas=1,
                n_gpus_per_replica=n_train,
            ),
            _rollout_group(),
        ]

    raise ValueError(f"unknown XRL_TOPOLOGY={TOPOLOGY!r} (use 'colocate' or 'disaggregate')")


SERVICE_GROUPS = build_service_groups()
COLOCATIONS = (
    [
        ColocationRing(
            group_id="actor_card",
            ring=(
                ("actor_infer", SchedulingMode.FALLBACK),
                ("actor_train", SchedulingMode.ON_DEMAND),
            ),
        )
    ]
    if TOPOLOGY == "colocate"
    else []
)


def main() -> None:
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
