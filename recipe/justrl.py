"""justrl recipe: GRPO for DeepSeek-R1-Distill-Qwen-1.5B on DAPO-Math-17k.

Asymmetric colocate on 8 cards, driven by the card-level SPMD Service stack:

* **inference**: 8 replicas x ``TP=1`` -- one single-card SGLang server per card;
* **training**: 1 replica x ``FSDP/DDP=8`` -- one 8-card TitanTrainer
  (``dp_replicate_degree=8``);
* **rollout**: 1 CPU-only rollout driver.

Both GPU sides share the same 8 physical cards. The training / inference
hyperparameters below are the frozen production config for this run and are not
meant to be env-tuned (unlike the ``grpo_gsm8k`` smoke recipe).

Run with the launcher (recommended)::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ENDPOINT=https://hf-mirror.com \\
        python scripts/launch.py --recipe recipe.justrl
"""

from __future__ import annotations

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

# ── Model / topology ────────────────────────────────────────────────────────
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
NUM_INFERENCE_ENGINES = 8  # one single-card (TP=1) SGLang server per card

# ── Rollout shape ───────────────────────────────────────────────────────────
ROLLOUT_BATCH = 256  # distinct prompts per step
GROUP_SIZE = 8  # GRPO completions per prompt
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # trainer trigger threshold: 2048
NUM_EPOCHS = 8

# ── Trajectory logging ──────────────────────────────────────────────────────
# When on, the rollout also dumps raw token ids / logprobs / masks alongside
# the human-readable trajectory / reward / advantage.
VERBOSE_TRAJECTORY_LOG = False


def _sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_new_tokens=15360,
    )


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name="qwen2_5_math",
        model_flavor="R1-Distill-1.5B",
        seq_len=16384,
        lr=1e-6,
        weight_decay=0.1,
        warmup_steps=0,
        max_norm=1.0,
        steps=3000,
        dtype="bfloat16",
        compile_model=True,
        dp_shard_degree=-1,
        dp_replicate_degree=8,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=True,
        checkpoint_folder="checkpoint",
        dump_folder="./outputs",
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
    # enable_memory_saver is required so /release_memory_occupation +
    # /resume_memory_occupation work for the colocate GPU hand-off.
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
    # Colocates onto the inference cards (asymmetric: 1x FSDP=8 vs 8x TP=1) and
    # waits for those engines to free GPU memory before loading.
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
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
