"""Four-GPU colocated JustRL recipe for MiniCPM5-2.6B.

Topology:

* inference: 4 single-card SGLang replicas;
* training: one 4-card TorchTitan trainer colocated with inference;
* rollout: one CPU-only rollout driver.

Run with the launcher::

    MINICPM5_LOCAL_PATH=/path/to/minicpm5-2.6b \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        python scripts/launch.py --recipe recipe.justrl_minicpm5_2_6b_4gpu

``MINICPM5_LOCAL_PATH`` is required and must point to the model checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

from meshy.config import (
    RolloutServiceConfig,
    InferenceServiceConfig,
    SamplingParams,
    TrainerConfig,
    TrainerParamsConfig,
    TrainingServiceConfig,
)
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor

try:
    MODEL_PATH = str(Path(os.environ["MINICPM5_LOCAL_PATH"]).expanduser().resolve())
except KeyError as exc:
    raise RuntimeError(
        "MINICPM5_LOCAL_PATH must be set to the MiniCPM5 checkpoint path"
    ) from exc
NUM_INFERENCE_ENGINES = 4

ROLLOUT_BATCH = 256
GROUP_SIZE = 8
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE
NUM_EPOCHS = 8

VERBOSE_TRAJECTORY_LOG = False


def _validate_local_model() -> None:
    model_dir = Path(MODEL_PATH)
    required_files = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    missing = [name for name in required_files if not (model_dir / name).is_file()]
    if not model_dir.is_dir() or missing or not any(model_dir.glob("*.safetensors")):
        detail = f"; missing files: {missing}" if missing else ""
        raise FileNotFoundError(
            f"MiniCPM5 must be loaded from a complete local model directory: "
            f"{model_dir}{detail}"
        )


def _sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_new_tokens=15360,
    )


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2.6B",
        seq_len=16384,
        lr=1e-6,
        weight_decay=0.1,
        warmup_steps=0,
        max_norm=1.0,
        steps=3000,
        dtype="bfloat16",
        compile_model=True,
        dp_shard_degree=-1,
        dp_replicate_degree=4,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=True,
        checkpoint_folder="checkpoint",
        dump_folder="./outputs/justrl_minicpm5_2_6b_4gpu",
    )


def _trainer_params() -> TrainerParamsConfig:
    return TrainerParamsConfig(
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
            dataset="meshy.dataset.math:MATH",
            dataset_kwargs={"batch_size": ROLLOUT_BATCH, "seed": 42},
            reward="meshy.dataset.math:MATH.reward",
            sampling_params=_sampling_params().as_dict(),
            group_size=GROUP_SIZE,
            poll_interval=2.0,
            pacing_window=1,
            num_epochs=NUM_EPOCHS,
            verbose_trajectory_log=VERBOSE_TRAJECTORY_LOG,
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
    _validate_local_model()
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
