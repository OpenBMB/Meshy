"""PPO + independent critic (VAPO-GAE) for MiniCPM5-2B at 128k, 4 nodes / 32 H800.

Scaled-up version of ``justrl_ii_minicpm5_2b_128k.py`` that matches the
full §3 configuration from ``justrl_ii_recipe.md``:

| §3 item              | value                          | here                            |
|---|---|---|
| nodes                | actor 2 · critic 2 (32×H800)  | 16 actor + 16 critic GPUs       |
| batch                | 60 prompts × 8 = GBS 480       | ROLLOUT_BATCH=60, GROUP_SIZE=8  |
| base model           | MiniCPM5-2B                    | ``openbmb/MiniCPM5-2B-SFT``     |
| dataset              | category-aligned S9, 19.6k     | ``S9_DATASET_PATH``             |
| seq_len / resp cap   | 131072 / 126976                | same                            |
| parallelism          | TP1 · CP4 · colocate sglang    | same per replica                |
| lr                   | 1e-6 constant, adam eps 1e-8   | same                            |
| clip-higher          | 0.2 / 0.28                     | same                            |
| TIS                  | on                             | ``use_tis=True``                |
| KL                   | none (kl_coef=0)               | no reference model              |
| critic cold start    | ~30 windows                    | ``COLD_START_WINDOWS``          |

Actor (16 GPUs, 2 nodes): 16 SGLang instances at TP1 colocated with a single
training replica; cp_degree=4 and dp_shard_degree=4 give a CP4/FSDP4 layout
across the four CP groups.

Critic (16 GPUs, 2 nodes): same CP4/FSDP4 layout on dedicated cards with no
ring membership, so the critic runs independently while inference and training
share the actor half.

The overlong-penalty caveat from the single-node recipe applies here too:
``external_advantage=True`` bypasses the rollout's advantage pipeline, so the
DAPO soft penalty is not yet wired. ``OVERLONG_BUFFER_LEN`` is recorded so
the wiring is a small change.

The backbone defaults to the Hub repo ``openbmb/MiniCPM5-2B-SFT``; set
``MINICPM5_LOCAL_PATH`` to a local checkout for ``HF_HUB_OFFLINE=1`` runs.

    S9_DATASET_PATH=/path/to/s9_no_advanced_math.jsonl \\
    HF_HOME=/path/to/hf-cache \\
    no_proxy="127.0.0.1,localhost,\\$no_proxy" \\
        python scripts/launch.py --recipe recipe.justrl_ii_minicpm5_2b_128k_32gpu
"""

from __future__ import annotations

import os
from pathlib import Path

from meshy.config import (
    GlobalCheckpointConfig,
    CriticServiceConfig,
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

#: Hub repo id by default; ``MINICPM5_LOCAL_PATH`` overrides it with a local
#: checkout (needed for ``HF_HUB_OFFLINE=1`` runs).
DEFAULT_MODEL = "openbmb/MiniCPM5-2B-SFT"
_LOCAL_MODEL = os.environ.get("MINICPM5_LOCAL_PATH")
MODEL_PATH = (
    str(Path(_LOCAL_MODEL).expanduser().resolve()) if _LOCAL_MODEL else DEFAULT_MODEL
)

try:
    DATASET_PATH = str(Path(os.environ["S9_DATASET_PATH"]).expanduser().resolve())
except KeyError as exc:
    raise RuntimeError("S9_DATASET_PATH must point to the S9 JSONL file") from exc

# ── §3: actor / critic split ─────────────────────────────────────────────
# 2 nodes × 8 GPUs each per side; CP4 within each side → 4 FSDP shards.
NUM_ACTOR_CARDS = 16
NUM_CRITIC_CARDS = 16

# ── §3: lengths ───────────────────────────────────────────────────────────
SEQ_LEN = 131072
MAX_NEW_TOKENS = 126976
OVERLONG_BUFFER_LEN = 25395  # §1; recorded, not yet wired -- see the docstring

# ── §3: batch ─────────────────────────────────────────────────────────────
# 60 prompts × 8 samples = GBS 480.
ROLLOUT_BATCH = 60
GROUP_SIZE = 8
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # 480
NUM_EPOCHS = 1
NUM_STEPS = 600

#: §2 cold start. The recipe uses 30; the env override lets a smoke run reach
#: the cold-start -> publish transition quickly.
COLD_START_WINDOWS = int(os.environ.get("XRL_COLD_START_WINDOWS", "30"))
CRITIC_SCORE_BATCH = int(os.environ.get("XRL_CRITIC_SCORE_BATCH", str(BATCH_SIZE)))

#: §3's "ckpt every 10 steps, rotated". One instance shared by the trainer,
#: the critic and the rollout -- the resume point is the newest window all
#: three committed a shard for. Resume with ``--runtime-dir <old run>
#: --resume``; see ``meshy/service/checkpoint.py``.
CHECKPOINT = GlobalCheckpointConfig(
    enabled=os.environ.get("XRL_CHECKPOINT", "1") not in ("0", "false", ""),
    interval=int(os.environ.get("XRL_CHECKPOINT_INTERVAL", "10")),
    keep_latest_k=int(os.environ.get("XRL_CHECKPOINT_KEEP", "3")),
)


def _validate_dataset() -> None:
    if not Path(DATASET_PATH).is_file():
        raise FileNotFoundError(f"S9 dataset does not exist: {DATASET_PATH}")


def _sampling_params() -> SamplingParams:
    # §3: T=1.0, top_p=1.0.
    return SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens=MAX_NEW_TOKENS
    )


def _actor_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2B",
        seq_len=SEQ_LEN,
        steps=NUM_STEPS,
        lr=1e-6,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.98,
        warmup_steps=0,
        # §3: constant 1e-6.
        lr_decay_ratio=0.0,
        # fp32 master weights; bf16 matmuls (mixed_precision_param default).
        dtype="float32",
        compile_model=True,
        activation_checkpoint_mode="full",
        # 16 actor GPUs: CP4 × FSDP4 (dp_shard) across 4 CP groups.
        dp_shard_degree=4,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=4,
        enable_checkpoint=True,
        checkpoint_folder="checkpoint",
        dump_folder="./outputs/justrl_ii_minicpm5_2b_128k_32gpu",
    )


def _critic_trainer_config() -> TrainerConfig:
    """Same architecture and parallel layout as the actor, on its own 16 cards.

    16 GPUs: CP4 × FSDP4 across 4 CP groups, mirroring the actor layout.
    GAE walks adjacent timesteps; ``predict_vapo_gae`` gathers across the CP
    group (``cp.py::gather_seq``), so CP4 here is load-bearing, not optional.
    """
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2B",
        seq_len=SEQ_LEN,
        max_norm=1.0,
        steps=NUM_STEPS,
        dtype="float32",
        compile_model=False,
        dp_shard_degree=4,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=4,
        # train_backbone=True keeps every layer's activations alive for the
        # backward, so full checkpointing is required regardless of seq_len.
        activation_checkpoint_mode="full",
        enable_checkpoint=False,
        dump_folder="./outputs/justrl_ii_minicpm5_2b_128k_32gpu/critic",
    )


def _trainer_params() -> TrainerParamsConfig:
    return TrainerParamsConfig(
        mini_batch_size=4,
        micro_batch_size=1,
        ppo_clip_eps_low=0.2,
        ppo_clip_eps_high=0.28,
        old_logprobs_source="rollout",
        calculate_per_token_loss=True,
        # §1 TIS: fuse against train/inference logprob mismatch.
        use_tis=True,
        tis_ratio_min=0.5,
        tis_ratio_max=5.0,
        logprob_chunk_size=2048,
    )


def _inference_group() -> ServiceGroup:
    # 16 SGLang instances at TP1, one per actor GPU.
    return ServiceGroup(
        id="actor_infer",
        n_replicas=NUM_ACTOR_CARDS,
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
        n_gpus_per_replica=NUM_ACTOR_CARDS,
        colocate_with="actor_infer",
        wait_until=["actor_infer"],
        config=TrainingServiceConfig(
            model_path=MODEL_PATH,
            trainer_config=_actor_trainer_config(),
            trainer_params=_trainer_params(),
            batch_size=BATCH_SIZE,
            timer_enabled=True,
            # Keeps the rollout gate monotonic across the cold-start hand-off.
            critic_cold_start_windows=COLD_START_WINDOWS,
            checkpoint=CHECKPOINT,
        ),
    )


def _critic_group() -> ServiceGroup:
    return ServiceGroup(
        id="critic",
        n_replicas=1,
        n_gpus_per_replica=NUM_CRITIC_CARDS,
        # Dedicated cards: no ring, no GPU arbitration with the actor.
        wait_until=["actor_infer"],
        config=CriticServiceConfig(
            # Backbone from the *base* model, never the actor's live weights.
            model_path=MODEL_PATH,
            trainer_config=_critic_trainer_config(),
            score_batch_size=CRITIC_SCORE_BATCH,
            # Keep the pre-``values`` arrangement: this critic reduces its
            # own GAE to one scalar per sequence and publishes ``advantage``,
            # so the trainer's default ``tq_fields`` still apply. See
            # ``recipe/justrl_ii_minicpm5_2b_128k.py`` for the split where
            # the critic publishes ``values`` and the trainer runs GAE.
            publish_mode="advantage",
            # §2: a frozen-backbone value head plateaus below the whitening
            # baseline; train_backbone=True is required for meaningful AUC.
            train_backbone=True,
            lr=5e-6,
            # §2: 10-iter warmup absorbs the cold-start value_loss spike (~32).
            warmup_steps=10,
            cold_start_windows=COLD_START_WINDOWS,
            value_epochs=1,
            gamma=1.0,
            # §1 VAPO: lambda_i = 1 - 1/(alpha * L_i). alpha=1.5 is calibrated
            # for 128k responses; the paper's 0.05 collapses the credit
            # half-life to ~500 tokens and causes entropy collapse at step ~55.
            alpha=1.5,
            max_norm=1.0,
            checkpoint=CHECKPOINT,
        ),
    )


def _rollout_group() -> ServiceGroup:
    return ServiceGroup(
        id="rollout",
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_train", "actor_infer", "critic"],
        config=RolloutServiceConfig(
            model_path=MODEL_PATH,
            dataset="meshy.dataset.s9_math:S9Math",
            dataset_kwargs={"path": DATASET_PATH, "batch_size": ROLLOUT_BATCH, "seed": 42},
            reward="meshy.dataset.s9_math:S9Math.reward",
            sampling_params=_sampling_params().as_dict(),
            group_size=GROUP_SIZE,
            num_epochs=NUM_EPOCHS,
            poll_interval=2.0,
            pacing_window=1,
            # The critic owns the advantage: the rollout neither computes nor
            # writes the column, which is what makes the trainer wait for it.
            external_advantage=True,
            # PPO+critic gives zero-variance groups a gradient (§1), so there
            # is no reason to drop them.
            filter_zero_std_groups=False,
            checkpoint=CHECKPOINT,
        ),
    )


def build_service_groups() -> list[ServiceGroup]:
    return [_inference_group(), _training_group(), _critic_group(), _rollout_group()]


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
    _validate_dataset()
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
