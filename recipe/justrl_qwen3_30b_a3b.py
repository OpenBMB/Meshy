"""justrl_qwen3_30b_a3b: JustRL(GRPO) with the Qwen3-30B-A3B MoE model.

The :mod:`recipe.justrl` pipeline with the model swapped for Qwen3-30B-A3B and
the colocate layout adapted to a 30B-parameter MoE on 8 cards:

* **inference**: 1 replica x ``TP=8 + EP=8`` -- justrl's 8x TP=1 layout cannot
  hold this model (60 GB of bf16 weights per single card), and plain TP=8 must
  not be used either: 30B-A3B is a fine-grained MoE (128 experts whose FFNs are
  only 768-dim), so TP would slice every expert into 96-dim slivers of GEMM.
  ``ep_size=8`` keeps attention tensor-parallel while distributing whole
  experts across the cards (16 intact 768-dim experts per GPU).
* **training**: 1 replica x ``FSDP=8`` (``dp_shard_degree=-1``) -- justrl's
  ``dp_replicate_degree=8`` would put the full 30B optimizer state on every
  card; sharding is mandatory at this size. ``compile_model`` is off (MoE).
* **rollout**: 1 CPU-only rollout driver, lock-step (``pacing_window=1``).

Both GPU sides share the same 8 physical cards; around every step the trainer
releases the TP=8 engine's memory, restores, trains, dumps the HF checkpoint
(~60 GB) and pushes it back via ``/update_weights_from_disk`` -- expect the
sync phase to be checkpoint-I/O bound at this size.

Generation length and training sequence length keep justrl's production values
(``max_new_tokens=15360``, ``seq_len=16384``) -- the thinking model needs the
full budget to finish its CoT and produce a scoreable boxed answer. The rollout
batch defaults to 32 prompts x 8 (256 samples/version), scaled from justrl's
256 x 8 for the single TP=8 engine; everything is env-tunable.

Run with the launcher::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ENDPOINT=https://hf-mirror.com \\
        python scripts/launch.py --recipe recipe.justrl_qwen3_30b_a3b
"""

from __future__ import annotations

import os

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
MODEL_PATH = os.environ.get("XRL_MODEL", "Qwen/Qwen3-30B-A3B")
MODEL_NAME = "qwen3"
MODEL_FLAVOR = "30B-A3B"
NGPUS = int(os.environ.get("XRL_NGPUS", "8"))

# ── Rollout shape ───────────────────────────────────────────────────────────
SEQ_LEN = int(os.environ.get("XRL_SEQ_LEN", "16384"))
MAX_NEW_TOKENS = int(os.environ.get("XRL_MAX_NEW_TOKENS", "15360"))
ROLLOUT_BATCH = int(os.environ.get("XRL_ROLLOUT_BATCH", "32"))  # prompts per step
GROUP_SIZE = int(os.environ.get("XRL_GROUP_SIZE", "8"))  # GRPO completions per prompt
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # trainer trigger threshold
NUM_EPOCHS = int(os.environ.get("XRL_EPOCHS", "1"))

# ── Trajectory logging ──────────────────────────────────────────────────────
VERBOSE_TRAJECTORY_LOG = False


def _sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name=MODEL_NAME,
        model_flavor=MODEL_FLAVOR,
        seq_len=SEQ_LEN,
        lr=float(os.environ.get("XRL_LR", "1e-6")),
        weight_decay=0.1,
        warmup_steps=0,
        max_norm=1.0,
        steps=int(os.environ.get("XRL_STEPS", "3000")),
        dtype="bfloat16",
        # torch.compile off: per-block compile is tuned for the dense models;
        # the MoE block's token routing recompiles pathologically.
        compile_model=False,
        # FSDP over all 8 cards (justrl's dp_replicate=8 cannot hold a 30B
        # model's full optimizer state per card).
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=False,
        dump_folder="./outputs/justrl_qwen3_30b_a3b",
    )


def _trainer_params() -> TrainerParamsConfig:
    return TrainerParamsConfig(
        # Conservative per-rank batching for the 30B MoE (justrl-1.5B used
        # mini_batch_size=8).
        mini_batch_size=1,
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
        n_replicas=1,
        n_gpus_per_replica=NGPUS,
        config=InferenceServiceConfig(
            model_path=MODEL_PATH,
            server_args={
                "model_path": MODEL_PATH,
                "tp_size": NGPUS,
                # Fine-grained MoE: distribute whole experts (EP) instead of
                # slicing each 768-dim expert FFN eight ways with TP.
                "ep_size": int(os.environ.get("XRL_EP_SIZE", str(NGPUS))),
                "enable_memory_saver": True,
                "mem_fraction_static": float(os.environ.get("XRL_MEM_FRACTION", "0.6")),
            },
        ),
    )


def _training_group() -> ServiceGroup:
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
