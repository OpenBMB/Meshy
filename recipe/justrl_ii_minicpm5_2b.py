"""PPO + independent critic (VAPO-GAE) for MiniCPM5-2B on one 8-card node.

The GRPO recipes derive the advantage from group-normalised reward, so a group
whose samples all pass (or all fail) carries no gradient. This recipe replaces
that with a separate PPO critic: advantage comes from a value baseline, so
zero-variance groups still carry signal (``justrl_ii_recipe.md`` §1).

Topology -- recipe §3 gives the actor and critic equal resources; scaled to a
single node that is 4 cards each::

    cards 0-3   actor_infer (4x SGLang)  +  actor_train (4-card FSDP)   [ring]
    cards 4-7   critic (4-card FSDP, dedicated -- no GPU arbitration)
    cpu         rollout

Data plane -- the rollout does not compute an advantage at all; the critic
publishes it, and TransferQueue's AND-filter is what orders the three roles::

    rollout --(no advantage)--> critic --(advantage)--> actor_train

Cold start (recipe §2) -- for the first ``COLD_START_WINDOWS`` windows the
critic trains alone and publishes nothing, so the actor never steps while the
cold-start value-loss spike is absorbed. The critic paces the rollout itself
during that phase.

The backbone is pulled from the Hub by default; set ``MINICPM5_LOCAL_PATH`` to
run against a local checkout instead.

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
        no_proxy=127.0.0.1,localhost,$no_proxy \\
        python scripts/launch.py --recipe recipe.justrl_ii_minicpm5_2b
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

#: Hub repo id by default -- transformers/SGLang resolve it through the usual
#: ``HF_HOME`` cache. ``MINICPM5_LOCAL_PATH`` overrides it with a local
#: checkout (needed for ``HF_HUB_OFFLINE=1`` runs).
DEFAULT_MODEL = "openbmb/MiniCPM5-2B-SFT"
_LOCAL_MODEL = os.environ.get("MINICPM5_LOCAL_PATH")
MODEL_PATH = (
    str(Path(_LOCAL_MODEL).expanduser().resolve()) if _LOCAL_MODEL else DEFAULT_MODEL
)

NUM_INFERENCE_ENGINES = 4
NUM_CRITIC_CARDS = 4

ROLLOUT_BATCH = 16
GROUP_SIZE = 8
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE
NUM_EPOCHS = 1

SEQ_LEN = 4096
MAX_NEW_TOKENS = 1024

#: Recipe §2 uses 30. Lower it to smoke the cold-start -> publish transition
#: quickly; raise it for a real run.
COLD_START_WINDOWS = int(os.environ.get("XRL_COLD_START_WINDOWS", "2"))
#: Rows per critic window. Matches the trainer's batch so the two stay in step.
CRITIC_SCORE_BATCH = BATCH_SIZE

#: Global checkpoint / resume. The *same* instance goes to the trainer, the
#: critic and the rollout: the resume point is the newest window all three
#: committed a shard for, so a service on its own interval would never
#: intersect (``build_topology`` rejects that).
#:
#: Cost at ``dtype="float32"``: ~9 GB of parameters plus ~18 GB of AdamW
#: moments per replica, actor and critic both, so roughly 54 GB per retained
#: step. Size the volume for ``keep_latest_k`` of those.
#:
#: Resume with::
#:
#:     python scripts/launch.py --recipe recipe.justrl_ii_minicpm5_2b \\
#:         --runtime-dir <the interrupted run's dir> --resume
CHECKPOINT = GlobalCheckpointConfig(
    enabled=os.environ.get("XRL_CHECKPOINT", "1") not in ("0", "false", ""),
    interval=int(os.environ.get("XRL_CHECKPOINT_INTERVAL", "10")),
    keep_latest_k=int(os.environ.get("XRL_CHECKPOINT_KEEP", "3")),
)


def _sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens=MAX_NEW_TOKENS
    )


def _actor_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2B",
        seq_len=SEQ_LEN,
        lr=1e-6,
        weight_decay=0.1,
        warmup_steps=0,
        max_norm=1.0,
        steps=3000,
        # fp32 master weights; bf16 matmuls (mixed_precision_param default).
        dtype="float32",
        compile_model=False,
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=False,
        dump_folder="./outputs/justrl_ii_minicpm5_2b",
    )


def _critic_trainer_config() -> TrainerConfig:
    """Same architecture as the actor, sharded over the critic's own cards.

    The critic is a separate replica: its own FSDP mesh, its own optimiser, no
    parameter shared with the actor.
    """
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2B",
        seq_len=SEQ_LEN,
        max_norm=1.0,
        steps=3000,
        # fp32 master weights; bf16 matmuls (see ``critic_coldstart_32k.py``).
        dtype="float32",
        compile_model=False,
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        # Recipe §3 runs CP4 for 128k context. At SEQ_LEN=4096 the critic
        # fits without it, and CP would only add gather/shard traffic around
        # every GAE recursion.
        cp_degree=1,
        # train_backbone keeps every layer's activations alive for the
        # backward, so the critic needs real checkpointing where the actor
        # can get away with less.
        activation_checkpoint_mode="full",
        enable_checkpoint=False,
        dump_folder="./outputs/justrl_ii_minicpm5_2b/critic",
    )


def _trainer_params() -> TrainerParamsConfig:
    return TrainerParamsConfig(
        mini_batch_size=8,
        micro_batch_size=1,
        # DAPO clip-higher (recipe §1)
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
            trainer_config=_actor_trainer_config(),
            trainer_params=_trainer_params(),
            batch_size=BATCH_SIZE,
            timer_enabled=True,
            # Keeps the rollout's gate stream monotonic across the hand-off
            # from the critic (which paces during cold start) to the trainer.
            critic_cold_start_windows=COLD_START_WINDOWS,
            checkpoint=CHECKPOINT,
        ),
    )


def _critic_group() -> ServiceGroup:
    return ServiceGroup(
        id="critic",
        n_replicas=1,
        n_gpus_per_replica=NUM_CRITIC_CARDS,
        # Dedicated cards: no colocate_with, so no ring membership and no GPU
        # arbitration. The critic runs while inference and training share the
        # other half of the node.
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
            train_backbone=True,
            lr=5e-6,
            warmup_steps=10,
            cold_start_windows=COLD_START_WINDOWS,
            value_epochs=1,
            gamma=1.0,
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
            dataset="meshy.dataset.math:MATH",
            dataset_kwargs={"batch_size": ROLLOUT_BATCH, "seed": 42},
            reward="meshy.dataset.math:MATH.reward",
            sampling_params=_sampling_params().as_dict(),
            group_size=GROUP_SIZE,
            poll_interval=1.0,
            pacing_window=1,
            num_epochs=NUM_EPOCHS,
            # The critic owns the advantage: the rollout neither computes nor
            # writes the column, which is what makes the trainer wait for it.
            external_advantage=True,
            # PPO+critic gives zero-variance groups a gradient (recipe §1), so
            # unlike the GRPO recipes there is no reason to drop them.
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
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
