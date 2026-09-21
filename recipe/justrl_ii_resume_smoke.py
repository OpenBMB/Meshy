"""Checkpoint / resume smoke test for the justrl_ii stack, on one 8-card node.

Everything about the topology matches ``recipe.justrl_ii_minicpm5_2b`` -- the
same four services, the same colocation ring, the same critic cold start ->
publish transition -- shrunk so a full global checkpoint lands every window
instead of every ten, and with a backbone small enough that a DCP shard is
seconds rather than minutes.

What it is for: the resume path has three independent halves (the actor's DCP
state, the critic's value net, the rollout's data cursor) plus a gate stream
that has to be re-established against an empty TransferQueue. Those only
interact in a live run.

Run it, kill it, resume it::

    # 1. run past a few checkpoints, then Ctrl-C (or SIGTERM the launcher)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
        no_proxy=127.0.0.1,localhost,$no_proxy \\
        python scripts/launch.py --recipe recipe.justrl_ii_resume_smoke \\
            --runtime-dir /tmp/justrl_ii_resume

    # 2. continue it
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
        no_proxy=127.0.0.1,localhost,$no_proxy \\
        python scripts/launch.py --recipe recipe.justrl_ii_resume_smoke \\
            --runtime-dir /tmp/justrl_ii_resume --resume

What to check in ``/tmp/justrl_ii_resume/logs`` after step 2:

* ``ignitor-0.log``   -- "resuming at global step N", with all three services
                         listed as participants;
* ``actor_train-0``   -- torchtitan's "Loading the checkpoint from
                         .../state/actor_train-0/step-N", and the next step
                         numbered N-COLD_START+1 rather than 1;
* ``critic-0``        -- "restored value net + optimiser", *not* "loaded ...
                         backbone params"; the cold start is not repeated;
* ``rollout-0``       -- "fast-forwarded to prompt X", and no
                         "gen_gate stream out of order" warning;
* ``weights/actor_train-0/`` -- only ``keep_latest_k`` versions plus whichever
                         ones the retained checkpoints name.

And the negative case: re-running step 1's command (same ``--runtime-dir``, no
``--resume``) must fail at startup rather than quietly continue.
"""

from __future__ import annotations

import os

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
from meshy.dataset.math import MATH
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor

MODEL_PATH = os.environ.get(
    "XRL_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

NUM_INFERENCE_ENGINES = 8
ROLLOUT_BATCH = 4
GROUP_SIZE = 4
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE
SEQ_LEN = 1024
MAX_NEW_TOKENS = 256

#: Two cold-start windows, so a kill can land either side of the critic ->
#: trainer hand-off and both resume paths get exercised.
COLD_START_WINDOWS = int(os.environ.get("XRL_COLD_START_WINDOWS", "2"))

#: Every window, and keep the minimum the store allows. A smoke run should hit
#: rotation within its first handful of steps.
CHECKPOINT = GlobalCheckpointConfig(
    enabled=True,
    interval=int(os.environ.get("XRL_CHECKPOINT_INTERVAL", "1")),
    keep_latest_k=int(os.environ.get("XRL_CHECKPOINT_KEEP", "2")),
)

#: Long enough that a resumed run has data left, short enough to end on its
#: own if nobody kills it.
NUM_PROMPT_BATCHES = int(os.environ.get("XRL_SMOKE_BATCHES", "40"))


class SmokeMATH(MATH):
    """MATH capped at a few batches, and seekable so a resume can skip ahead.

    ``MATH`` already carries the plain integer ``index`` the resume contract
    accepts; the only thing added here is the cap, whose own counter has to
    move with the cursor or a resumed run would get ``NUM_PROMPT_BATCHES``
    *more* batches on top of the ones it already consumed.
    """

    def __init__(self, batch_size: int, seed: int | None = None, **kwargs):
        super().__init__(batch_size=batch_size, seed=seed, **kwargs)
        self._limit = NUM_PROMPT_BATCHES * batch_size

    def seek(self, index: int) -> None:
        self.index = int(index)

    def next_batch(self, builder):
        if self.index >= self._limit:
            return []
        return super().next_batch(builder)


def _sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens=MAX_NEW_TOKENS
    )


def _trainer_config(dump: str) -> TrainerConfig:
    return TrainerConfig(
        model_name="qwen2_5_math",
        model_flavor="R1-Distill-1.5B",
        seq_len=SEQ_LEN,
        lr=1e-6,
        max_norm=1.0,
        warmup_steps=0,
        steps=1000,
        dtype="bfloat16",
        compile_model=False,
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=1,
        activation_checkpoint_mode="full",
        dump_folder=dump,
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
            trainer_config=_trainer_config("./outputs/justrl_ii_resume_smoke"),
            trainer_params=TrainerParamsConfig(
                mini_batch_size=4,
                micro_batch_size=1,
                ppo_clip_eps_low=0.2,
                ppo_clip_eps_high=0.28,
                old_logprobs_source="train",
            ),
            batch_size=BATCH_SIZE,
            timer_enabled=True,
            critic_cold_start_windows=COLD_START_WINDOWS,
            # Small enough to force rotation early, large enough to outlast
            # the pacing window (1 here).
            weights_keep_latest_k=3,
            checkpoint=CHECKPOINT,
        ),
    )


def _critic_group() -> ServiceGroup:
    return ServiceGroup(
        id="critic",
        n_replicas=1,
        n_gpus_per_replica=NUM_INFERENCE_ENGINES,
        colocate_with="actor_infer",
        wait_until=["actor_infer"],
        config=CriticServiceConfig(
            model_path=MODEL_PATH,
            trainer_config=_trainer_config("./outputs/justrl_ii_resume_smoke/critic"),
            score_batch_size=BATCH_SIZE,
            publish_mode="advantage",
            micro_rows=1,
            train_backbone=True,
            lr=5e-6,
            warmup_steps=2,
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
            dataset="recipe.justrl_ii_resume_smoke:SmokeMATH",
            dataset_kwargs={"batch_size": ROLLOUT_BATCH, "seed": 42},
            reward="meshy.dataset.math:MATH.reward",
            sampling_params=_sampling_params().as_dict(),
            group_size=GROUP_SIZE,
            poll_interval=1.0,
            pacing_window=1,
            num_epochs=1,
            external_advantage=True,
            filter_zero_std_groups=False,
            checkpoint=CHECKPOINT,
        ),
    )


SERVICE_GROUPS = [
    _inference_group(),
    _training_group(),
    _critic_group(),
    _rollout_group(),
]
COLOCATIONS = [
    ColocationRing(
        group_id="actor_card",
        ring=(
            ("actor_infer", SchedulingMode.FALLBACK),
            ("critic", SchedulingMode.ON_DEMAND),
            ("actor_train", SchedulingMode.ON_DEMAND),
        ),
    )
]


def main() -> None:
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
