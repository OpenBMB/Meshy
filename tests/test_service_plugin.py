"""Tests for the self-describing ServiceConfig plugin contract."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from meshy.config import ServiceConfig
from meshy.service.base import GPU, Service, ServiceGroup
from meshy.service.registry import resolve_service
from meshy.service.topology import build_topology


@dataclass
class CustomGPUConfig(ServiceConfig):
    role: ClassVar[str] = "custom_gpu"
    service_cls: ClassVar[str] = "meshy.service.base:Service"
    endpoint_port_base: ClassVar[int] = 33000
    dist_port_base: ClassVar[int] = 43000


@dataclass
class CustomCPUConfig(ServiceConfig):
    role: ClassVar[str] = "custom_cpu"
    service_cls: ClassVar[str] = "meshy.service.base:Service"
    uses_gpu: ClassVar[bool] = False


def _gpu() -> GPU:
    return GPU(host="127.0.0.1", global_rank=0, node_rank=0, local_rank=0)


def test_custom_service_resolves_without_registry_entry():
    assert resolve_service(CustomGPUConfig()) is Service


def test_custom_gpu_and_cpu_services_enter_generic_topology():
    topology = build_topology(
        [
            ServiceGroup(id="compute", config=CustomGPUConfig()),
            ServiceGroup(
                id="driver",
                config=CustomCPUConfig(),
                n_gpus_per_replica=0,
            ),
        ],
        [_gpu()],
    )

    assert topology.by_name("compute-0").endpoint_port == 33000
    assert topology.by_name("compute-0").dist_port == 43000
    assert [service.name for service in topology.gpu_services()] == ["compute-0"]
    assert [service.name for service in topology.cpu_services()] == ["driver-0"]
    assert topology.services_by_role("custom_cpu")[0].name == "driver-0"


def test_custom_gpu_service_must_declare_port_bases():
    @dataclass
    class MissingPortsConfig(ServiceConfig):
        role: ClassVar[str] = "missing_ports"
        service_cls: ClassVar[str] = "meshy.service.base:Service"

    with pytest.raises(ValueError, match="must declare endpoint_port_base"):
        build_topology(
            [ServiceGroup(id="broken", config=MissingPortsConfig())],
            [_gpu()],
        )


def test_colocated_custom_service_must_use_distinct_port_bases():
    @dataclass
    class CollidingConfig(ServiceConfig):
        role: ClassVar[str] = "colliding"
        service_cls: ClassVar[str] = "meshy.service.base:Service"
        endpoint_port_base: ClassVar[int] = 33000
        dist_port_base: ClassVar[int] = 43000

    with pytest.raises(ValueError, match="endpoint_port collision"):
        build_topology(
            [
                ServiceGroup(id="first", config=CustomGPUConfig()),
                ServiceGroup(
                    id="second",
                    config=CollidingConfig(),
                    colocate_with="first",
                ),
            ],
            [_gpu()],
        )


def test_service_config_must_declare_implementation():
    @dataclass
    class MissingImplementationConfig(ServiceConfig):
        role: ClassVar[str] = "missing_implementation"

    with pytest.raises(ValueError, match="must declare service_cls"):
        resolve_service(MissingImplementationConfig())
