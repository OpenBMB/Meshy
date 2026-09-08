"""Student Top-K on-policy distillation (OPD) on GSM8K.

One Teacher (Titan/FSDP, frozen) scores every rollout row with its Top-K token
ids and log-probs; the Student trainer minimises the forward KL over that
support.  Rollout and inference are the stock GRPO services: the Teacher only
appends two TransferQueue columns to the rows the rollout already writes, and
the Student fetches the union.

Run with the launcher::

    python scripts/launch.py --recipe recipe.student_topk_opd

Or directly::

    XRL_TOPOLOGY=disaggregate XRL_NGPUS=4 \\
        torchrun --standalone --nproc-per-node 4 -m recipe.student_topk_opd

Topologies (``XRL_TOPOLOGY``):
* ``colocate``     -- inference, Teacher and Student time-share all cards
  through one colocation ring (inference is the fallback owner).
* ``disaggregate`` -- Teacher on its own ``XRL_TEACHER_GPUS`` cards, the rest
  split between standalone inference replicas and the Student trainer.
"""

from __future__ import annotations

import os

from meshy.config import (
    InferenceServiceConfig,
    OPDTeacherConfig,
    OPDTrainingConfig,
    RolloutServiceConfig,
    TrainerConfig,
    TrainerParamsConfig,
)
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor

STUDENT_PATH = os.environ.get("XRL_MODEL", "Qwen/Qwen3-1.7B")
STUDENT_NAME = os.environ.get("XRL_MODEL_NAME", "qwen3")
STUDENT_FLAVOR = os.environ.get("XRL_MODEL_FLAVOR", "1.7B")
TEACHER_PATH = os.environ.get("XRL_TEACHER", "Qwen/Qwen3-8B")
TEACHER_NAME = os.environ.get("XRL_TEACHER_NAME", "qwen3")
TEACHER_FLAVOR = os.environ.get("XRL_TEACHER_FLAVOR", "8B")
NGPUS = int(os.environ.get("XRL_NGPUS", "1"))
TOPOLOGY = os.environ.get("XRL_TOPOLOGY", "colocate")
SEQ_LEN = int(os.environ.get("XRL_SEQ_LEN", "2048"))
MAX_NEW_TOKENS = int(os.environ.get("XRL_MAX_NEW_TOKENS", "1024"))
TOP_K = int(os.environ.get("XRL_TOP_K", "8"))

ROLLOUT_BATCH = int(os.environ.get("XRL_ROLLOUT_BATCH", "4"))
GROUP_SIZE = int(os.environ.get("XRL_GROUP_SIZE", "4"))
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # trainer trigger threshold
# Rows per Teacher forward window; must divide BATCH_SIZE so the trainer's
# fetch never waits on a partial window.
SCORE_BATCH = int(os.environ.get("XRL_SCORE_BATCH", str(BATCH_SIZE)))
assert BATCH_SIZE % SCORE_BATCH == 0, "XRL_SCORE_BATCH must divide the training batch"


def _trainer_config(name: str, flavor: str, dump: str) -> TrainerConfig:
    return TrainerConfig(
        model_name=name,
        model_flavor=flavor,
        seq_len=SEQ_LEN,
        steps=int(os.environ.get("XRL_STEPS", "1000")),
        dtype="bfloat16",
        lr=float(os.environ.get("XRL_LR", "1e-6")),
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=False,
        dump_folder=dump,
        compile_model=False,
    )


def _inference_config(tp_size: int, *, enable_memory_saver: bool) -> InferenceServiceConfig:
    server_args = {"model_path": STUDENT_PATH, "tp_size": tp_size}
    if enable_memory_saver:
        server_args["enable_memory_saver"] = True
    return InferenceServiceConfig(model_path=STUDENT_PATH, server_args=server_args)


def _teacher_config() -> OPDTeacherConfig:
    return OPDTeacherConfig(
        model_path=TEACHER_PATH,
        trainer_config=_trainer_config(TEACHER_NAME, TEACHER_FLAVOR, "./outputs/student_topk_opd/teacher"),
        top_k=TOP_K,
        score_batch_size=SCORE_BATCH,
    )


def _training_config() -> OPDTrainingConfig:
    return OPDTrainingConfig(
        model_path=STUDENT_PATH,
        trainer_config=_trainer_config(STUDENT_NAME, STUDENT_FLAVOR, "./outputs/student_topk_opd/student"),
        trainer_params=TrainerParamsConfig(mini_batch_size=1, micro_batch_size=1),
        batch_size=BATCH_SIZE,
        timer_enabled=True,
    )


def _rollout_group() -> ServiceGroup:
    return ServiceGroup(
        id="rollout",
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_train", "actor_infer", "teacher"],
        config=RolloutServiceConfig(
            model_path=STUDENT_PATH,
            dataset="meshy.dataset.gsm8k:GSM8K",
            dataset_kwargs={"batch_size": ROLLOUT_BATCH, "split": "train", "seed": 42},
            reward="meshy.dataset.gsm8k:GSM8K.reward",
            sampling_params={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": MAX_NEW_TOKENS},
            group_size=GROUP_SIZE,
            poll_interval=2.0,
            pacing_window=1,
            num_epochs=int(os.environ.get("XRL_EPOCHS", "1")),
        ),
    )


def build_service_groups() -> list[ServiceGroup]:
    if TOPOLOGY == "colocate":
        return [
            ServiceGroup(id="actor_infer", config=_inference_config(NGPUS, enable_memory_saver=True),
                         n_replicas=1, n_gpus_per_replica=NGPUS),
            ServiceGroup(id="teacher", config=_teacher_config(), n_replicas=1, n_gpus_per_replica=NGPUS,
                         colocate_with="actor_infer", wait_until=["actor_infer"]),
            ServiceGroup(id="actor_train", config=_training_config(), n_replicas=1, n_gpus_per_replica=NGPUS,
                         colocate_with="actor_infer", wait_until=["actor_infer"]),
            _rollout_group(),
        ]

    if TOPOLOGY == "disaggregate":
        n_teacher = int(os.environ.get("XRL_TEACHER_GPUS", "1"))
        rest = NGPUS - n_teacher
        if rest < 2:
            raise ValueError("disaggregate topology needs XRL_NGPUS >= XRL_TEACHER_GPUS + 2")
        n_train = int(os.environ.get("XRL_TRAIN_GPUS", str(max(1, rest // 2))))
        n_infer = rest - n_train
        if n_infer < 1:
            raise ValueError("disaggregate needs at least 1 inference card")
        return [
            ServiceGroup(id="actor_infer", config=_inference_config(1, enable_memory_saver=False),
                         n_replicas=n_infer, n_gpus_per_replica=1),
            ServiceGroup(id="teacher", config=_teacher_config(), n_replicas=1, n_gpus_per_replica=n_teacher),
            ServiceGroup(id="actor_train", config=_training_config(), n_replicas=1, n_gpus_per_replica=n_train),
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
                ("teacher", SchedulingMode.ON_DEMAND),
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
