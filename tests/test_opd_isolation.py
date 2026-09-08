"""The OPD role is opt-in: configs plug into the generic topology and the
GRPO main line never imports the OPD modules."""

import subprocess
import sys

import pytest

from meshy.config import (
    GRPO_TRAINER_FIELDS,
    OPD_TRAINER_FIELDS,
    OPDTeacherConfig,
    OPDTrainingConfig,
    TrainerConfig,
)
from meshy.service.opd import OPDTeacherService, OPDTrainingService
from meshy.service.base import GPU, ServiceGroup
from meshy.service.registry import resolve_service
from meshy.service.topology import build_topology


def _gpus(n):
    return [GPU(host="127.0.0.1", global_rank=i, node_rank=0, local_rank=i) for i in range(n)]


def test_trainer_fields_extend_grpo_fields():
    assert list(OPD_TRAINER_FIELDS[: len(GRPO_TRAINER_FIELDS)]) == list(GRPO_TRAINER_FIELDS)
    assert OPDTrainingConfig(model_path="m", trainer_config=TrainerConfig(), batch_size=4).tq_fields == list(OPD_TRAINER_FIELDS)


def test_configs_resolve_to_role_services_and_validate():
    assert resolve_service(OPDTeacherConfig(model_path="t", trainer_config=TrainerConfig())) is OPDTeacherService
    assert resolve_service(OPDTrainingConfig(model_path="m", trainer_config=TrainerConfig(), batch_size=4)) is OPDTrainingService
    with pytest.raises(ValueError, match="top_k"):
        OPDTeacherConfig(model_path="t", trainer_config=TrainerConfig(), top_k=0)
    with pytest.raises(ValueError, match="context parallelism"):
        OPDTeacherConfig(model_path="t", trainer_config=TrainerConfig(cp_degree=2))


def test_teacher_enters_generic_topology_as_training_peer():
    groups = [
        ServiceGroup(id="teacher", config=OPDTeacherConfig(model_path="t", trainer_config=TrainerConfig()),
                     n_replicas=1, n_gpus_per_replica=2),
        ServiceGroup(id="actor_train",
                     config=OPDTrainingConfig(model_path="m", trainer_config=TrainerConfig(), batch_size=4),
                     n_replicas=1, n_gpus_per_replica=2),
    ]
    topology = build_topology(groups, _gpus(4))
    assert [s.name for s in topology.services_by_role("teacher")] == ["teacher-0"]
    assert [s.name for s in topology.training_services()] == ["actor_train-0"]
    assert topology.by_name("teacher-0").endpoint_port == 32000


@pytest.mark.parametrize("topology,ngpus", [("colocate", 2), ("disaggregate", 4)])
def test_recipe_builds_both_topologies(topology, ngpus):
    code = (
        "import os; os.environ['XRL_TOPOLOGY']=%r; os.environ['XRL_NGPUS']=%r\n"
        "from recipe import student_topk_opd as r\n"
        "from meshy.service.topology import build_topology\n"
        "from meshy.service.base import GPU\n"
        "t = build_topology(r.SERVICE_GROUPS, [GPU(host='h', global_rank=i, node_rank=0, local_rank=i) for i in range(%d)])\n"
        "print(sorted(s.role for s in t.services))\n"
    ) % (topology, str(ngpus), ngpus)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "'teacher'" in out and "'training'" in out and "'rollout'" in out and "'inference'" in out


def test_grpo_main_line_never_imports_opd():
    code = (
        "import sys, os; os.environ['XRL_NGPUS']='2'\n"
        "import recipe.grpo_gsm8k, meshy.service.training, meshy.service.rollout, "
        "meshy.worker.rollout, meshy.worker.titan, meshy.engine.titan, meshy.backend.titan\n"
        "print([m for m in sys.modules if m.endswith('.opd') or m.endswith('.topk_loss')])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out
