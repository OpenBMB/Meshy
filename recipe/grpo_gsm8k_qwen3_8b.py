"""GRPO-on-GSM8K for **Qwen3-8B**, colocate on 8 cards (asymmetric layout).

Inference and training share the *same* 8 physical cards, but partitioned
differently:

* **inference**: 8 replicas x ``TP=1`` — one single-card SGLang server per card,
  for high-throughput generation;
* **training**: 1 replica x ``FSDP=8`` — one 8-card fully-sharded TitanTrainer.

This is an *asymmetric* colocate group (8x1 inference vs 1x8 training); both
sides occupy the same 8 cards. Around every training step the trainer's rank 0
releases all 8 inference engines' GPU memory, restores the sharded trainer,
trains one step, offloads the trainer, then resumes + weight-syncs all 8
inference engines from the freshly dumped HF checkpoint.

Run via the launcher (a single ``torchrun``; each shared card ignites its
inference replica first, then the colocate trainer)::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
    XRL_MODEL=<Qwen3-8B local path or HF id> HF_ENDPOINT=https://hf-mirror.com \\
        python scripts/launch.py --recipe recipe.grpo_gsm8k_qwen3_8b

Everything is env-tunable (model path, seq len, batch shape, ...); see the
module constants below.
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

MODEL_PATH = os.environ.get("XRL_MODEL", "Qwen/Qwen3-8B")
MODEL_NAME = os.environ.get("XRL_MODEL_NAME", "qwen3")
MODEL_FLAVOR = os.environ.get("XRL_MODEL_FLAVOR", "8B")
# Cards shared by the 8x TP=1 inference engines and the single FSDP trainer.
NGPUS = int(os.environ.get("XRL_NGPUS", "8"))
SEQ_LEN = int(os.environ.get("XRL_SEQ_LEN", "2048"))
MAX_NEW_TOKENS = int(os.environ.get("XRL_MAX_NEW_TOKENS", "1024"))

ROLLOUT_BATCH = int(os.environ.get("XRL_ROLLOUT_BATCH", "32"))  # prompts per step
GROUP_SIZE = int(os.environ.get("XRL_GROUP_SIZE", "8"))  # completions per prompt
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # trainer trigger threshold


def _trainer_config() -> TrainerConfig:
    # dp_shard_degree=-1 + tp_degree=1 => FSDP sharded over all NGPUS cards.
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
        dump_folder="./outputs/grpo_gsm8k_qwen3_8b",
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


def _inference_group() -> ServiceGroup:
    # One single-card SGLang server per card. enable_memory_saver is required so
    # /release_memory_occupation + /resume_memory_occupation work for the
    # colocate GPU hand-off.
    return ServiceGroup(
        id="actor_infer",
        n_replicas=NGPUS,
        n_gpus_per_replica=1,
        config=InferenceServiceConfig(
            model_path=MODEL_PATH,
            server_args={
                "model_path": MODEL_PATH,
                "tp_size": 1,
                "enable_memory_saver": True,
            },
        ),
    )


def _training_group() -> ServiceGroup:
    # Asymmetric colocate: single FSDP=NGPUS trainer sharing the 8x TP=1
    # inference cards; waits for inference to free GPU memory before loading.
    return ServiceGroup(
        id="actor_train",
        n_replicas=1,
        n_gpus_per_replica=NGPUS,
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
