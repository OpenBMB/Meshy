"""Lazy Service class resolution.

The formal plugin path lives on each :class:`~meshy.config.ServiceConfig`
subclass. ``SERVICE_TYPES`` remains as a compatibility lookup for callers that
only have a built-in role string; new Services do not register in it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meshy.config import ServiceConfig
    from meshy.service.base import Service

SERVICE_TYPES: dict[str, str] = {
    "inference": "meshy.service.inference:SGLangService",
    "training": "meshy.service.training:TitanTrainingService",
    "rollout": "meshy.service.rollout:RolloutService",
}


def resolve_service(config_or_role: "ServiceConfig | str") -> "type[Service]":
    """Resolve a config's declared Service class, or a legacy built-in role."""
    if isinstance(config_or_role, str):
        try:
            path = SERVICE_TYPES[config_or_role]
        except KeyError:
            raise ValueError(
                f"unknown service role {config_or_role!r}; "
                f"known roles: {sorted(SERVICE_TYPES)}"
            ) from None
    else:
        config_type = type(config_or_role)
        path = config_type.service_cls
        if not path:
            raise ValueError(
                f"ServiceConfig {config_type.__name__} must declare service_cls"
            )
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(
            f"invalid service_cls {path!r}; expected 'package.module:ClassName'"
        )
    service_type = getattr(importlib.import_module(module_name), attr)
    from meshy.service.base import Service

    if not isinstance(service_type, type) or not issubclass(service_type, Service):
        raise TypeError(f"service_cls {path!r} does not resolve to a Service subclass")
    return service_type
