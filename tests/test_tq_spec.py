"""derive_tq_spec: the launcher's fallback TransferQueue sizing.

TransferQueue is mandatory infrastructure for every recipe, so a recipe that
exports no ``TRANSFER_QUEUE`` dict must still get a sound spec derived from its
typed service configs.
"""

from __future__ import annotations

from meshy.config import (
    RolloutServiceConfig,
    InferenceServiceConfig,
    TrainingServiceConfig,
    TrainerConfig,
)
from meshy.service.base import ServiceGroup
from meshy.transferqueue.spec import derive_tq_spec


def _groups(batch_size: int, max_running: int) -> list[ServiceGroup]:
    return [
        ServiceGroup(id="actor_infer", config=InferenceServiceConfig(model_path="m")),
        ServiceGroup(
            id="actor_train",
            config=TrainingServiceConfig(
                model_path="m", trainer_config=TrainerConfig(), batch_size=batch_size
            ),
        ),
        ServiceGroup(
            id="rollout",
            n_gpus_per_replica=0,
            config=RolloutServiceConfig(
                model_path="m",
                dataset="d:D",
                async_max_running_request=max_running,
            ),
        ),
    ]


def test_spec_scales_with_batch_size(tmp_path, monkeypatch):
    from meshy.transferqueue.client import store_ref

    monkeypatch.delenv("XRL_TQ_PRE_ALLOC", raising=False)
    monkeypatch.delenv("XRL_TQ_ENDPOINTS", raising=False)
    spec = derive_tq_spec(_groups(batch_size=2048, max_running=-1), str(tmp_path))
    assert spec["pre_alloc_sample_num"] == 4 * 2048
    # Default discovery is the bootstrap-store ref; XRL_TQ_ENDPOINTS overrides.
    assert spec["endpoints_file"] == store_ref(str(tmp_path))
    assert spec["num_storage_units"] >= 1


def test_spec_scales_with_async_in_flight(tmp_path, monkeypatch):
    monkeypatch.delenv("XRL_TQ_PRE_ALLOC", raising=False)
    spec = derive_tq_spec(_groups(batch_size=64, max_running=3072), str(tmp_path))
    assert spec["pre_alloc_sample_num"] == 2 * 3072 + 64


def test_spec_has_a_floor_for_tiny_smoke_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("XRL_TQ_PRE_ALLOC", raising=False)
    spec = derive_tq_spec(_groups(batch_size=4, max_running=-1), str(tmp_path))
    assert spec["pre_alloc_sample_num"] == 1024


def test_env_overrides_win(tmp_path, monkeypatch):
    monkeypatch.setenv("XRL_TQ_PRE_ALLOC", "77")
    monkeypatch.setenv("XRL_TQ_STORAGE_UNITS", "5")
    spec = derive_tq_spec(_groups(batch_size=2048, max_running=-1), str(tmp_path))
    assert spec["pre_alloc_sample_num"] == 77
    assert spec["num_storage_units"] == 5
