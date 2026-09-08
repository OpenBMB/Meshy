from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx

from meshy.engine.sglang import SGLangEngine
from meshy.engine.titan import _publish_weights_to_inference
from meshy.config import InferenceServiceConfig, TrainingServiceConfig
from meshy.service.base import GPU, ServiceGroup
from meshy.service.colocation import GpuGrant
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.topology import build_topology
from meshy.worker.titan import TitanWorker


def test_publisher_updates_only_inference_outside_colocation_ring(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    def post(url: str, *, json: dict, timeout: float):
        calls.append((url, json))
        assert timeout == 1800.0
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    _publish_weights_to_inference(
        [
            {"name": "actor-infer-0", "endpoint": "http://infer:30000"},
            {"name": "standalone-infer-0", "endpoint": "http://infer:30001/"},
        ],
        "/runtime/weights/titan/v3",
        3,
        "titan",
        managed_names={"actor-infer-0"},
    )

    assert calls == [
        (
            "http://infer:30001/update_weights_from_disk",
            {"model_path": "/runtime/weights/titan/v3"},
        )
    ]


def test_sglang_colocate_acquire_uses_checkpoint_from_grant() -> None:
    engine = SGLangEngine(["http://infer:30000"])
    restored: list[str] = []
    engine.model_path = "/models/base"
    engine.restore_for_colocate = restored.append  # type: ignore[method-assign]
    try:
        engine.on_colocate_acquire(
            GpuGrant("cards", 4, "titan", "actor-infer", payload_ref="/weights/v4")
        )
    finally:
        asyncio.run(engine.close())

    assert restored == ["/weights/v4"]


def test_sglang_non_genesis_acquire_rejects_missing_checkpoint() -> None:
    engine = SGLangEngine(["http://infer:30000"])
    engine.model_path = "/models/base"
    try:
        try:
            engine.on_colocate_acquire(
                GpuGrant("cards", 4, "titan", "actor-infer", payload_ref=None)
            )
        except RuntimeError as exc:
            assert "latest checkpoint" in str(exc)
        else:
            raise AssertionError("missing checkpoint path was accepted")
    finally:
        asyncio.run(engine.close())


def test_titan_worker_attaches_checkpoint_to_colocation_release() -> None:
    class Colocation:
        def __init__(self) -> None:
            self.releases: list[dict] = []

        def release(self, **kwargs) -> None:
            self.releases.append(kwargs)

    worker = TitanWorker.__new__(TitanWorker)
    worker.engine = SimpleNamespace(
        name="titan",
        step_index=2,
        weight_version=2,
        stream_minibatch=False,
        step=lambda samples, sync: SimpleNamespace(
            step=3, weights_path="/weights/v3"
        ),
    )
    worker.colocation = Colocation()
    worker._colocation_request = object()
    worker.batch_size = 1
    worker.trained_since_sync = 0
    worker.stream_minibatch = False
    worker._gate_output = lambda *, step: {"gate": step}

    assert worker.process_tq_batch([SimpleNamespace()]) == {"gate": 3}
    assert worker.colocation.releases == [
        {"transition": "train-step-complete", "payload_ref": "/weights/v3"}
    ]
    assert worker._colocation_request is None


def test_colocation_elects_one_manager_per_ring_member_group() -> None:
    groups = [
        ServiceGroup(
            id="actor-infer",
            n_replicas=2,
            n_gpus_per_replica=1,
            config=InferenceServiceConfig(model_path="/models/base"),
        ),
        ServiceGroup(
            id="actor-train",
            n_replicas=1,
            n_gpus_per_replica=2,
            colocate_with="actor-infer",
            config=TrainingServiceConfig(
                model_path="/models/base",
                trainer_config=SimpleNamespace(),
                batch_size=1,
            ),
        ),
    ]
    topology = build_topology(
        groups,
        [GPU("127.0.0.1", 0, 0, 0), GPU("127.0.0.1", 1, 0, 1)],
        [
            ColocationRing(
                "actor-card",
                (("actor-infer", SchedulingMode.FALLBACK), ("actor-train", SchedulingMode.ON_DEMAND)),
                poll_interval=0.001,
            )
        ],
    )

    assert [s.is_colocation_leader for s in topology.group("actor-infer")] == [True, False]
    assert [s.is_colocation_leader for s in topology.group("actor-train")] == [True]
