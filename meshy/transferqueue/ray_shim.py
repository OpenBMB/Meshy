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

"""A minimal, *local-execution* mock of the ``ray`` module.

TransferQueue imports ``ray`` at module-load time and decorates its Controller and
storage-unit classes with ``@ray.remote``.  However, on the default ``SimpleStorage``
backend Ray is used **only** for bootstrap / discovery / lifecycle — never on the
data hot path.  Once components exchange their ZMQ endpoints (``ZMQServerInfo``),
all control and data traffic flows over ZMQ.

This shim replaces ``ray`` with local semantics so TransferQueue runs without Ray
installed:

* ``@ray.remote`` / ``@ray.remote(...)`` wraps a class so that ``.remote(...)``
  instantiates it **in-process** as a plain object (its ``__init__`` already starts
  its own ZMQ server threads).
* ``handle.method.remote(*a, **k)`` calls the underlying instance method synchronously
  and returns the result directly.
* ``ray.get(x)`` is the identity function (results are already concrete).
* ``ray.get_actor(name, ...)`` looks up a process-local named-actor registry.

Install it by calling :func:`install` *before* importing ``transfer_queue``.
"""

from __future__ import annotations

import socket
import sys
import types
from typing import Any

__all__ = ["install", "is_installed"]

# Process-local registry of named actors, mirroring Ray's named-actor namespace.
# Keyed by (namespace, name) -> LocalActorHandle.
_NAMED_ACTORS: dict[tuple[str | None, str], "LocalActorHandle"] = {}


# --------------------------------------------------------------------------- #
# Actor emulation
# --------------------------------------------------------------------------- #
class _RemoteMethod:
    """Wraps a bound method so that ``.remote(*a, **k)`` calls it synchronously."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    # Allow direct calls too, just in case.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


class LocalActorHandle:
    """Handle around a plain in-process instance, mimicking a Ray actor handle.

    Attribute access returns a :class:`_RemoteMethod` so that
    ``handle.some_method.remote(...)`` works exactly like Ray, but executes locally.
    """

    def __init__(self, instance: Any, *, name: str | None = None, namespace: str | None = None) -> None:
        # Use object.__setattr__ to avoid recursing through __getattr__.
        object.__setattr__(self, "_tq_instance", instance)
        object.__setattr__(self, "_tq_name", name)
        object.__setattr__(self, "_tq_namespace", namespace)

    def __getattr__(self, item: str) -> _RemoteMethod:
        instance = object.__getattribute__(self, "_tq_instance")
        attr = getattr(instance, item)
        if callable(attr):
            return _RemoteMethod(attr)
        # Non-callable attributes: wrap in a trivial .remote accessor as well.
        return _RemoteMethod(lambda _value=attr: _value)

    def _tq_kill(self) -> None:
        """Tear down the underlying instance, mimicking Ray's force-kill semantics.

        Ray's ``ray.kill`` abruptly terminates the actor *process*; it does NOT run
        the actor's graceful GC finalizers in the caller. We mirror that:

        * Detach any ``weakref.finalize`` finalizer the instance registered. This is
          important — TransferQueue's ``SimpleStorageUnit`` registers a finalizer that
          calls ``zmq.Context.term()``, which blocks indefinitely (the upstream
          shutdown path never closes its inproc worker socket before terminating).
          Under real Ray this never runs; running it here would hang ``close()``.
        * Signal the instance's shutdown event so its polling daemon threads exit.

        All of TransferQueue's background threads are daemon threads, so anything left
        running is reaped cleanly at interpreter exit. We intentionally do not call
        ``term()`` or close sockets from this (foreign) thread, which would be unsafe
        while the instance's proxy/worker threads are still using them.
        """
        instance = object.__getattribute__(self, "_tq_instance")
        name = object.__getattribute__(self, "_tq_name")
        namespace = object.__getattribute__(self, "_tq_namespace")

        # Remove from named registry so a same-named actor can be recreated.
        if name is not None:
            _NAMED_ACTORS.pop((namespace, name), None)

        # Prevent the blocking GC finalizer from ever running.
        finalizer = getattr(instance, "_finalizer", None)
        if finalizer is not None:
            try:
                finalizer.detach()
            except Exception:
                pass

        # Ask polling daemon threads to stop on their next tick.
        event = getattr(instance, "_shutdown_event", None)
        if event is not None:
            try:
                event.set()
            except Exception:
                pass


class LocalActorClass:
    """Result of ``@ray.remote`` on a class.

    Mimics Ray's ``ActorClass``: supports ``.options(...)`` and ``.remote(...)``,
    and otherwise transparently proxies to the wrapped class (so static attributes
    and ``isinstance`` checks against it still behave reasonably).
    """

    def __init__(self, cls: type, options: dict[str, Any] | None = None) -> None:
        self._tq_cls = cls
        self._tq_options = dict(options or {})

    def options(self, **kwargs: Any) -> "LocalActorClass":
        merged = dict(self._tq_options)
        merged.update(kwargs)
        return LocalActorClass(self._tq_cls, merged)

    def remote(self, *args: Any, **kwargs: Any) -> LocalActorHandle:
        name = self._tq_options.get("name")
        namespace = self._tq_options.get("namespace")

        # Mimic Ray: creating a second actor with an existing name raises ValueError.
        if name is not None and (namespace, name) in _NAMED_ACTORS:
            raise ValueError(
                f"An actor with name '{name}' in namespace '{namespace}' already exists."
            )

        instance = self._tq_cls(*args, **kwargs)
        handle = LocalActorHandle(instance, name=name, namespace=namespace)

        if name is not None:
            _NAMED_ACTORS[(namespace, name)] = handle
        return handle

    # Transparent proxy to the wrapped class for attribute access.
    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_tq_cls"), item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Ray forbids calling an actor class directly; we allow plain instantiation
        # to be forgiving, but steer callers toward .remote().
        return self._tq_cls(*args, **kwargs)


def remote(*args: Any, **kwargs: Any) -> Any:
    """Mock of ``ray.remote``.

    Supports both bare ``@ray.remote`` and parameterized ``@ray.remote(num_cpus=1)``
    usage, applied to either classes or functions.
    """

    def _wrap(obj: Any) -> Any:
        if isinstance(obj, type):
            return LocalActorClass(obj)
        # Functions: expose a .remote that runs synchronously.
        method = _RemoteMethod(obj)
        # Preserve direct callability of the original function.
        method.__call__ = obj  # type: ignore[method-assign]
        return method

    # Bare decorator: @ray.remote
    if len(args) == 1 and not kwargs and (isinstance(args[0], type) or callable(args[0])):
        return _wrap(args[0])

    # Parameterized: @ray.remote(num_cpus=1)
    def _decorator(obj: Any) -> Any:
        return _wrap(obj)

    return _decorator


# --------------------------------------------------------------------------- #
# Object / task API
# --------------------------------------------------------------------------- #
class ObjectRef:  # noqa: D401 - placeholder type for annotations only
    """Placeholder for ``ray.ObjectRef`` (used only in type annotations)."""


def get(refs: Any, *, timeout: float | None = None) -> Any:
    """Identity ``ray.get`` — results from this shim are already concrete."""
    if isinstance(refs, list):
        return [get(r) for r in refs]
    return refs


def put(value: Any, **kwargs: Any) -> Any:
    """Identity ``ray.put`` — store nothing, return the value itself."""
    return value


def wait(refs: list, *, num_returns: int = 1, timeout: float | None = None) -> tuple[list, list]:
    ready = list(refs[:num_returns])
    remaining = list(refs[num_returns:])
    return ready, remaining


def get_actor(name: str, namespace: str | None = None) -> LocalActorHandle:
    """Look up a named actor in the process-local registry.

    Raises ``ValueError`` when not found, matching Ray's behavior (TransferQueue's
    ``_init_from_existing`` relies on this).
    """
    handle = _NAMED_ACTORS.get((namespace, name))
    if handle is None:
        raise ValueError(f"Failed to look up actor with name '{name}' in namespace '{namespace}'.")
    return handle


def kill(handle: Any) -> None:
    """Tear down a local actor handle."""
    if isinstance(handle, LocalActorHandle):
        handle._tq_kill()


def is_initialized() -> bool:
    return True


def init(*args: Any, **kwargs: Any) -> None:
    """No-op ``ray.init`` (kept for compatibility with callers that invoke it)."""
    return None


def shutdown(*args: Any, **kwargs: Any) -> None:
    return None


def nodes() -> list:
    return []


def cancel(*args: Any, **kwargs: Any) -> None:
    return None


# --------------------------------------------------------------------------- #
# Runtime context (used by storage/managers/base.py to size a thread pool)
# --------------------------------------------------------------------------- #
class _RuntimeContext:
    def get_actor_id(self) -> None:
        return None

    def get_task_id(self) -> None:
        return None

    def get_assigned_resources(self) -> dict:
        return {}


def get_runtime_context() -> _RuntimeContext:
    return _RuntimeContext()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _detect_node_ip() -> str:
    """Best-effort local IP detection, replacing ``ray.util.get_node_ip_address``.

    Uses a UDP socket trick to find the primary outbound interface address, falling
    back to the loopback address if no network is reachable.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # No packet is actually sent for a UDP connect; it just picks a route.
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


# --------------------------------------------------------------------------- #
# Module assembly / installation
# --------------------------------------------------------------------------- #
def _build_util_module() -> types.ModuleType:
    util = types.ModuleType("ray.util")

    def get_node_ip_address() -> str:
        return _detect_node_ip()

    class _PlacementGroup:
        """Dummy placement group; SPREAD/STRICT_SPREAD have no meaning in-process."""

        def ready(self) -> Any:
            return None

    def placement_group(*args: Any, **kwargs: Any) -> _PlacementGroup:
        return _PlacementGroup()

    def remove_placement_group(*args: Any, **kwargs: Any) -> None:
        return None

    util.get_node_ip_address = get_node_ip_address  # type: ignore[attr-defined]
    util.placement_group = placement_group  # type: ignore[attr-defined]
    util.remove_placement_group = remove_placement_group  # type: ignore[attr-defined]
    return util


def _build_exceptions_module() -> types.ModuleType:
    exceptions = types.ModuleType("ray.exceptions")

    class RayError(Exception):
        pass

    class GetTimeoutError(RayError):
        pass

    class RayActorError(RayError):
        pass

    exceptions.RayError = RayError  # type: ignore[attr-defined]
    exceptions.GetTimeoutError = GetTimeoutError  # type: ignore[attr-defined]
    exceptions.RayActorError = RayActorError  # type: ignore[attr-defined]
    return exceptions


def _build_ray_module() -> types.ModuleType:
    ray = types.ModuleType("ray")
    ray.__version__ = "0.0.0+tq-rayless-shim"  # type: ignore[attr-defined]

    # Actor / object / task API
    ray.remote = remote  # type: ignore[attr-defined]
    ray.ObjectRef = ObjectRef  # type: ignore[attr-defined]
    ray.get = get  # type: ignore[attr-defined]
    ray.put = put  # type: ignore[attr-defined]
    ray.wait = wait  # type: ignore[attr-defined]
    ray.get_actor = get_actor  # type: ignore[attr-defined]
    ray.kill = kill  # type: ignore[attr-defined]
    ray.is_initialized = is_initialized  # type: ignore[attr-defined]
    ray.init = init  # type: ignore[attr-defined]
    ray.shutdown = shutdown  # type: ignore[attr-defined]
    ray.nodes = nodes  # type: ignore[attr-defined]
    ray.cancel = cancel  # type: ignore[attr-defined]
    ray.get_runtime_context = get_runtime_context  # type: ignore[attr-defined]

    # Submodules
    util = _build_util_module()
    exceptions = _build_exceptions_module()
    ray.util = util  # type: ignore[attr-defined]
    ray.exceptions = exceptions  # type: ignore[attr-defined]

    return ray


def is_installed() -> bool:
    mod = sys.modules.get("ray")
    return bool(mod is not None and getattr(mod, "__version__", "").endswith("tq-rayless-shim"))


def install(force: bool = False) -> types.ModuleType:
    """Install the fake ``ray`` module into ``sys.modules``.

    Must be called *before* importing ``transfer_queue``. Idempotent.

    Args:
        force: Reinstall even if a (real or shim) ``ray`` is already present.

    Returns:
        The installed shim ``ray`` module.
    """
    existing = sys.modules.get("ray")
    if existing is not None and not force:
        if is_installed():
            return existing
        raise RuntimeError(
            "A real 'ray' module is already imported. Install the shim earlier "
            "(before importing ray/transfer_queue), or pass force=True."
        )

    ray = _build_ray_module()
    sys.modules["ray"] = ray
    sys.modules["ray.util"] = ray.util  # type: ignore[attr-defined]
    sys.modules["ray.exceptions"] = ray.exceptions  # type: ignore[attr-defined]
    return ray
