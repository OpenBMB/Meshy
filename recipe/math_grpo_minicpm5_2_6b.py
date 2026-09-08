"""4-GPU 128k MiniCPM5 GRPO recipe for the local JSONL dataset.

Run with ``MINICPM5_LOCAL_PATH=/path/to/model S9_DATASET_PATH=/path/to/s9.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch.py --recipe recipe.math_grpo_minicpm5_2_6b``.
"""

from __future__ import annotations

import os
from pathlib import Path

from meshy.config import (
    InferenceServiceConfig,
    RolloutServiceConfig,
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
    raise RuntimeError("MINICPM5_LOCAL_PATH must be set") from exc

try:
    DATASET_PATH = str(Path(os.environ["S9_DATASET_PATH"]).expanduser().resolve())
except KeyError as exc:
    raise RuntimeError("S9_DATASET_PATH must point to the S9 JSONL file") from exc

NUM_GPUS = 8
SEQ_LEN = 131072
MAX_NEW_TOKENS = 126976
# Original 8-card recipe values were per-card; 4 cards → 4× those totals.
ROLLOUT_BATCH = 256
GROUP_SIZE = 8
BATCH_SIZE = 512
NUM_STEPS = 600


def _sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2.6B",
        seq_len=SEQ_LEN,
        steps=NUM_STEPS,
        lr=1e-6,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.98,
        warmup_steps=0,
        # Constant 1e-6 like the MiniCPM4-8B Miles baseline (run 673223).
        # torchtitan's default would decay linearly to 0 over ``steps``.
        lr_decay_ratio=0.0,
        dtype="bfloat16",
        compile_model=True,
        # 32k tokens/rank x 42 layers does not fit alongside the 130k-vocab
        # LM head under per-op SAC.
        activation_checkpoint_mode="full",
        dp_shard_degree=1,
        dp_replicate_degree=2,
        tp_degree=1,
        cp_degree=4,
        enable_checkpoint=True,
        checkpoint_folder="checkpoint",
        dump_folder="./outputs/justrl_minicpm5_2_6b_s9_long",
    )


def _trainer_params() -> TrainerParamsConfig:
    return TrainerParamsConfig(
        mini_batch_size=4,
        micro_batch_size=1,
        ppo_clip_eps_low=0.2,
        ppo_clip_eps_high=0.28,
        old_logprobs_source="rollout",
        calculate_per_token_loss=True,
        use_tis=True,
        tis_ratio_min=0.5,
        tis_ratio_max=5.0,
        logprob_chunk_size=2048,
    )


def _inference_group() -> ServiceGroup:
    return ServiceGroup(
        id="actor_infer",
        n_replicas=NUM_GPUS,
        n_gpus_per_replica=1,
        config=InferenceServiceConfig(
            model_path=MODEL_PATH,
            server_args={
                "model_path": MODEL_PATH,
                "tp_size": 1,
                "attention_backend": "fa3",
                "mem_fraction_static": 0.88,
                "max_running_requests": 64,
                "max_total_tokens": 1440000,
                "schedule_conservativeness": 1.2,
                "enable_memory_saver": True,
            },
        ),
    )


def _training_group() -> ServiceGroup:
    return ServiceGroup(
        id="actor_train",
        n_replicas=1,
        n_gpus_per_replica=NUM_GPUS,
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
            dataset="meshy.dataset.s9_math:S9Math",
            dataset_kwargs={"path": DATASET_PATH, "batch_size": ROLLOUT_BATCH, "seed": 42},
            reward="meshy.dataset.s9_math:S9Math.reward",
            advantage="meshy.advantage:_2_6_math_reshaped_advantage",
            advantage_kwargs={
                "rollout_max_response_len": MAX_NEW_TOKENS,
                "overlong_buffer_len": 25395,
                "overlong_penalty_factor": 1.0,
                "length_reward_weight": 0.2,
                "length_reward_min_spread": 8000,
                "length_reward_budget_floor": 10000,
            },
            filter_zero_std_groups=True,
            sampling_params=_sampling_params().as_dict(),
            group_size=GROUP_SIZE,
            num_epochs=1,
            async_max_running_request=1024,
            pacing_window=None,
            poll_interval=2.0,
        ),
    )


SERVICE_GROUPS = [_inference_group(), _training_group(), _rollout_group()]
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
