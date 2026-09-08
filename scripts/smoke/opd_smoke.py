"""Bounded smoke run of :mod:`recipe.student_topk_opd`.

Same topology as the recipe (defaults: 2 colocated cards), Qwen3-1.7B Student
and Qwen3-4B Teacher, two prompt batches so the run covers two full
rollout -> Teacher -> train -> weight-sync cycles and then exits::

    CUDA_VISIBLE_DEVICES=1,2 python scripts/launch.py \\
        --recipe scripts.smoke.opd_smoke --runtime-dir /tmp/opd_smoke

``XRL_SMOKE_INFER_PORT`` / ``XRL_SMOKE_TRAIN_PORT`` / ``XRL_SMOKE_TRAIN_DIST_PORT``
move the inference and trainer port bases away from the recipe defaults when
a stale server from an earlier run is still holding them.
"""

from __future__ import annotations

import dataclasses
import os

for _k, _v in {
    "XRL_TOPOLOGY": "colocate", "XRL_NGPUS": "2", "XRL_ROLLOUT_BATCH": "2", "XRL_GROUP_SIZE": "2",
    "XRL_MAX_NEW_TOKENS": "256", "XRL_SEQ_LEN": "1024", "XRL_TEACHER": "Qwen/Qwen3-4B",
    "XRL_TEACHER_FLAVOR": "4B", "XRL_SCORE_BATCH": "2", "XRL_STEPS": "20",
}.items():
    os.environ.setdefault(_k, _v)

from meshy.config import InferenceServiceConfig, OPDTrainingConfig  # noqa: E402
from meshy.dataset.gsm8k import GSM8K  # noqa: E402
from recipe import student_topk_opd as base  # noqa: E402

SMOKE_BATCHES = int(os.environ.get("XRL_SMOKE_BATCHES", "2"))


@dataclasses.dataclass
class SmokeInferenceConfig(InferenceServiceConfig):
    endpoint_port_base = int(os.environ.get("XRL_SMOKE_INFER_PORT", "30100"))


@dataclasses.dataclass
class SmokeTrainingConfig(OPDTrainingConfig):
    endpoint_port_base = int(os.environ.get("XRL_SMOKE_TRAIN_PORT", "31100"))
    dist_port_base = int(os.environ.get("XRL_SMOKE_TRAIN_DIST_PORT", "41100"))


class SmokeGSM8K(GSM8K):
    def __init__(self, batch_size: int, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
        self._batches_left = SMOKE_BATCHES

    def next_batch(self, builder):
        if self._batches_left <= 0:
            return []
        self._batches_left -= 1
        return super().next_batch(builder)


def _smoke(group):
    if group.id == "actor_infer":
        return dataclasses.replace(group, config=SmokeInferenceConfig(
            model_path=group.config.model_path, server_args=group.config.server_args))
    if group.id == "actor_train":
        return dataclasses.replace(group, config=SmokeTrainingConfig(**{f.name: getattr(group.config, f.name) for f in dataclasses.fields(group.config)}))
    if group.id == "rollout":
        return dataclasses.replace(group, config=dataclasses.replace(
            group.config, dataset="scripts.smoke.opd_smoke:SmokeGSM8K"))
    return group


SERVICE_GROUPS = [_smoke(g) for g in base.SERVICE_GROUPS]
COLOCATIONS = base.COLOCATIONS


def main() -> None:
    from meshy.service.ignite import Ignitor

    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
