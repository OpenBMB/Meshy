# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launcher utilities for running TransferQueue without Ray.

Two deployment modes are supported:

**Single-process (default, full-featured).** :func:`start_local` simply calls
``transfer_queue.init`` under the ray shim, which instantiates the controller and
N storage units in-process (each running its own ZMQ server threads). All public
APIs — KV, low-level client, StreamingDataLoader — then work as usual.

**Multi-machine (optional).** Each component is started as a standalone process via
the ``tq-controller`` / ``tq-storage`` CLI commands, which publish their ZMQ endpoints
to a shared JSON *endpoints file*. Clients then call :func:`connect` (or
:func:`build_config`) to read that file and connect directly over ZMQ — no Ray and no
``init()`` involved. This mirrors the ray-free client construction TransferQueue
already does internally in ``StreamingDataset._create_client``.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid importing transfer_queue at module import time
    from omegaconf import DictConfig
    from transfer_queue import TransferQueueClient


# --------------------------------------------------------------------------- #
# Single-process mode
# --------------------------------------------------------------------------- #
def start_local(conf: "DictConfig | dict | None" = None) -> "DictConfig | None":
    """Start a full TransferQueue system in the current process.

    Thin wrapper over ``transfer_queue.init``. The ray shim makes the controller and
    storage units run as plain in-process objects with their own ZMQ servers.

    Args:
        conf: Optional config to merge over TransferQueue's packaged ``config.yaml``
              (e.g. ``{"backend": {"SimpleStorage": {"num_data_storage_units": 4}}}``).

    Returns:
        The merged configuration produced by ``init`` (contains the populated
        ``controller.zmq_info`` and ``backend.SimpleStorage.zmq_info``).
    """
    import transfer_queue as tq
    from omegaconf import OmegaConf

    if conf is not None and not _is_dictconfig(conf):
        conf = OmegaConf.create(conf, flags={"allow_objects": True})
    return tq.init(conf)


# --------------------------------------------------------------------------- #
# ZMQServerInfo (de)serialization
# --------------------------------------------------------------------------- #
def server_info_to_jsonable(info: Any) -> dict:
    """Convert a ``ZMQServerInfo`` to a plain JSON-serializable dict."""
    role = info.role
    # Role is an enum whose value is a str; normalize to its string value.
    role_value = getattr(role, "value", role)
    return {
        "role": str(role_value),
        "id": info.id,
        "ip": info.ip,
        "ports": dict(info.ports),
    }


def server_info_from_jsonable(data: dict) -> Any:
    """Rebuild a ``ZMQServerInfo`` from a dict produced by :func:`server_info_to_jsonable`."""
    from transfer_queue.utils.enum_utils import Role
    from transfer_queue.utils.zmq_utils import ZMQServerInfo

    role = Role(data["role"])
    return ZMQServerInfo(
        role=role,
        id=data["id"],
        ip=data["ip"],
        ports={k: int(v) for k, v in data["ports"].items()},
    )


# --------------------------------------------------------------------------- #
# Endpoints file (multi-machine discovery)
# --------------------------------------------------------------------------- #
def _read_endpoints(path: str) -> dict:
    if not os.path.exists(path):
        return {"controller": None, "storage": {}}
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"controller": None, "storage": {}}


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON atomically so concurrent readers never see a half-written file."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".endpoints-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def publish_controller(path: str, info: Any) -> None:
    """Record the controller's ZMQ endpoint into the endpoints file."""
    data = _read_endpoints(path)
    data["controller"] = server_info_to_jsonable(info)
    _atomic_write_json(path, data)


def publish_storage_unit(path: str, rank: int, info: Any) -> None:
    """Record a storage unit's ZMQ endpoint (keyed by rank) into the endpoints file.

    NOTE: with multiple machines writing concurrently, point them at a shared file
    on a networked filesystem, or collect each unit's JSON fragment and merge once.
    """
    data = _read_endpoints(path)
    storage = data.get("storage") or {}
    storage[str(rank)] = server_info_to_jsonable(info)
    data["storage"] = storage
    _atomic_write_json(path, data)


# --------------------------------------------------------------------------- #
# Bootstrap-store publication (file-less discovery)
# --------------------------------------------------------------------------- #
def _store_client(addr: str, timeout_s: float = 60.0):
    """Connect to a ``torch.distributed.TCPStore`` at ``host:port`` as a client."""
    from datetime import timedelta

    from torch.distributed import TCPStore

    host, _, port = addr.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"--publish-store expects 'host:port', got {addr!r}")
    return TCPStore(host, int(port), is_master=False, timeout=timedelta(seconds=timeout_s))


def publish_to_store(addr: str, key: str, info: Any) -> None:
    """Publish a component's ``ZMQServerInfo`` under ``key`` in a TCPStore.

    Unlike the endpoints file, every publisher writes its *own* key, so there is
    no read-modify-write race and no serialization requirement between
    concurrently starting components.
    """
    _store_client(addr).set(key, json.dumps(server_info_to_jsonable(info)))


# --------------------------------------------------------------------------- #
# Building a config / client from an endpoints file
# --------------------------------------------------------------------------- #
def endpoints_to_config(
    data: dict, base_conf: "DictConfig | dict | None" = None, source: str = "<endpoints>"
) -> "DictConfig":
    """Build a TransferQueue config from an endpoints payload dict.

    ``data`` has the same shape as the endpoints file: ``{"controller": {...},
    "storage": {"0": {...}, ...}}`` -- regardless of whether it came from a
    file, a bootstrap store, or was assembled in-process.
    """
    from importlib import resources

    from omegaconf import OmegaConf

    if data.get("controller") is None:
        raise RuntimeError(
            f"No controller endpoint found in {source!r}. "
            "Start `tq-controller` first."
        )
    storage = data.get("storage") or {}
    if not storage:
        raise RuntimeError(
            f"No storage units found in {source!r}. "
            "Start `tq-storage --rank <i> --size <n>` for each unit."
        )

    conf = OmegaConf.create({}, flags={"allow_objects": True})
    default_conf = OmegaConf.load(resources.files("transfer_queue") / "config.yaml")
    conf = OmegaConf.merge(conf, default_conf)
    if base_conf is not None:
        if not _is_dictconfig(base_conf):
            base_conf = OmegaConf.create(base_conf, flags={"allow_objects": True})
        conf = OmegaConf.merge(conf, base_conf)

    # Reconstruct ZMQServerInfo objects and inject them.
    conf.controller.zmq_info = server_info_from_jsonable(data["controller"])

    backend_name = conf.backend.storage_backend
    if backend_name != "SimpleStorage":
        raise NotImplementedError(
            f"Multi-machine launcher currently supports only SimpleStorage, got {backend_name!r}."
        )
    storage_infos = {
        f"TransferQueueStorageUnit#{rank}": server_info_from_jsonable(info)
        for rank, info in sorted(storage.items(), key=lambda kv: int(kv[0]))
    }
    conf.backend[backend_name].zmq_info = storage_infos
    return conf


def build_config(endpoints_file: str, base_conf: "DictConfig | dict | None" = None) -> "DictConfig":
    """Build a TransferQueue config from a published endpoints file.

    The result is equivalent to what ``init()`` stores: ``controller.zmq_info`` and
    ``backend.SimpleStorage.zmq_info`` are populated with reconstructed
    ``ZMQServerInfo`` objects, ready to construct a client directly.

    Args:
        endpoints_file: Path to the JSON endpoints file written by the
                        ``tq-controller`` / ``tq-storage`` commands.
        base_conf: Optional overrides merged on top of TransferQueue's packaged
                   ``config.yaml`` (e.g. a custom sampler name, backend tweaks).

    Returns:
        A ``DictConfig`` suitable for :func:`connect` or for constructing a
        ``TransferQueueClient`` / ``StreamingDataset`` directly.
    """
    return endpoints_to_config(_read_endpoints(endpoints_file), base_conf, source=endpoints_file)


def connect_config(conf: "DictConfig") -> "TransferQueueClient":
    """Construct a ``TransferQueueClient`` from an already-built config."""
    import os as _os

    from transfer_queue import TransferQueueClient

    backend_name = conf.backend.storage_backend
    client = TransferQueueClient(
        client_id=f"TransferQueueClient_{_os.getpid()}",
        controller_info=conf.controller.zmq_info,
    )
    client.initialize_storage_manager(manager_type=backend_name, config=conf.backend[backend_name])
    return client


def connect_endpoints(
    data: dict, base_conf: "DictConfig | dict | None" = None, source: str = "<endpoints>"
) -> "TransferQueueClient":
    """Construct a ready-to-use client from an endpoints payload dict."""
    return connect_config(endpoints_to_config(data, base_conf, source=source))


def connect(endpoints_file: str, base_conf: "DictConfig | dict | None" = None) -> "TransferQueueClient":
    """Construct a ready-to-use ``TransferQueueClient`` from an endpoints file.

    This does NOT call ``init()`` — it builds the client directly over ZMQ (the
    process-local named-actor registry does not span machines, so discovery goes
    through the endpoints file instead).

    Args:
        endpoints_file: Path to the published endpoints file.
        base_conf: Optional config overrides (see :func:`build_config`).

    Returns:
        A connected ``TransferQueueClient``.
    """
    return connect_config(build_config(endpoints_file, base_conf))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_dictconfig(obj: Any) -> bool:
    try:
        from omegaconf import DictConfig

        return isinstance(obj, DictConfig)
    except Exception:
        return False
