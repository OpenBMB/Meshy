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

"""Command-line entry points for running TransferQueue components standalone.

These commands let you run a controller and storage units as independent OS
processes (on one or several machines), discovering each other via a shared JSON
endpoints file:

    # machine A
    tq-controller --endpoints-file /shared/ep.json

    # machine B (one process per storage unit)
    tq-storage --endpoints-file /shared/ep.json --rank 0 --size 100000
    tq-storage --endpoints-file /shared/ep.json --rank 1 --size 100000

    # any client process
    python -c "import tq_rayless as tq; c = tq.connect('/shared/ep.json'); ..."

Importing this module installs the ray shim (via ``import tq_rayless``) before any
``transfer_queue`` import, so no Ray is required.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

# Importing the package installs the ray shim and makes transfer_queue importable.
from meshy.transferqueue import tq_rayless  # noqa: F401
from meshy.transferqueue import launcher


def _block_until_signal(banner: str) -> None:
    """Print a readiness banner and block until SIGINT/SIGTERM."""
    stop = threading.Event()

    def _handler(signum, frame):  # noqa: ANN001
        stop.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    print(banner, flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        print("Shutting down.", flush=True)


def _add_publish_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoints-file", default=None, help="Path to the shared JSON endpoints file to publish to.")
    parser.add_argument("--publish-store", default=None, metavar="HOST:PORT",
                        help="Publish to a torch.distributed.TCPStore at this address instead of (or in addition to) a file.")
    parser.add_argument("--publish-key", default="tq",
                        help="Key prefix inside the store (default: 'tq'). The component publishes to '<prefix>/controller' or '<prefix>/storage/<rank>'.")


def _validate_publish_args(parser: argparse.ArgumentParser, args) -> None:
    if not args.endpoints_file and not args.publish_store:
        parser.error("one of --endpoints-file / --publish-store is required")


def _publish_targets(args, component_key: str, info, rank: int | None = None) -> list[str]:
    """Publish ``info`` to every configured target; return their descriptions."""
    targets = []
    if args.endpoints_file:
        if rank is None:
            launcher.publish_controller(args.endpoints_file, info)
        else:
            launcher.publish_storage_unit(args.endpoints_file, rank, info)
        targets.append(args.endpoints_file)
    if args.publish_store:
        key = f"{args.publish_key}/{component_key}"
        launcher.publish_to_store(args.publish_store, key, info)
        targets.append(f"store {args.publish_store} key {key!r}")
    return targets


def controller_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tq-controller", description="Run a standalone TransferQueue controller.")
    _add_publish_args(parser)
    parser.add_argument("--sampler", default="SequentialSampler", help="Sampler class name (default: SequentialSampler).")
    parser.add_argument("--polling-mode", action="store_true", help="Return empty BatchMeta instead of blocking when data is insufficient.")
    args = parser.parse_args(argv)
    _validate_publish_args(parser, args)

    from transfer_queue.controller import TransferQueueController
    from transfer_queue import sampler as sampler_mod

    sampler_cls = getattr(sampler_mod, args.sampler, None)
    if sampler_cls is None:
        parser.error(f"Unknown sampler {args.sampler!r}. Available: {[n for n in dir(sampler_mod) if n.endswith('Sampler')]}")

    # The @ray.remote shim turns this into an in-process actor handle.
    handle = TransferQueueController.options(  # type: ignore[attr-defined]
        name="TransferQueueController", namespace="transfer_queue"
    ).remote(sampler=sampler_cls(), polling_mode=args.polling_mode)

    info = handle.get_zmq_server_info.remote()
    targets = _publish_targets(args, "controller", info)

    _block_until_signal(
        f"[tq-controller] ready: id={info.id} ip={info.ip} ports={dict(info.ports)}\n"
        f"[tq-controller] published to {', '.join(targets)}"
    )
    return 0


def storage_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tq-storage", description="Run a standalone TransferQueue SimpleStorage unit.")
    _add_publish_args(parser)
    parser.add_argument("--rank", type=int, required=True, help="Unique storage-unit rank (0-based). Must be contiguous across all units.")
    parser.add_argument("--size", type=int, default=100000, help="Max number of samples this unit can hold (default: 100000).")
    args = parser.parse_args(argv)
    _validate_publish_args(parser, args)

    from transfer_queue.storage.simple_storage import SimpleStorageUnit

    handle = SimpleStorageUnit.options(  # type: ignore[attr-defined]
        name=f"TransferQueueStorageUnit#{args.rank}"
    ).remote(storage_unit_size=args.size)

    info = handle.get_zmq_server_info.remote()
    targets = _publish_targets(args, f"storage/{args.rank}", info, rank=args.rank)

    _block_until_signal(
        f"[tq-storage] ready: rank={args.rank} id={info.id} ip={info.ip} ports={dict(info.ports)}\n"
        f"[tq-storage] published to {', '.join(targets)}"
    )
    return 0


def info_main(argv: list[str] | None = None) -> int:
    """Print the contents of an endpoints file in a readable form."""
    parser = argparse.ArgumentParser(prog="tq-info", description="Show TransferQueue endpoints from a published file.")
    parser.add_argument("endpoints_file", help="Path to the JSON endpoints file.")
    args = parser.parse_args(argv)

    data = launcher._read_endpoints(args.endpoints_file)
    controller = data.get("controller")
    storage = data.get("storage") or {}

    print(f"Endpoints file: {args.endpoints_file}")
    if controller:
        print(f"  controller: id={controller['id']} ip={controller['ip']} ports={controller['ports']}")
    else:
        print("  controller: <not registered>")
    if storage:
        for rank in sorted(storage, key=int):
            su = storage[rank]
            print(f"  storage#{rank}: id={su['id']} ip={su['ip']} ports={su['ports']}")
    else:
        print("  storage: <none registered>")
    return 0


if __name__ == "__main__":  # pragma: no cover
    # Allow `python -m tq_rayless.cli controller ...` style invocation.
    if len(sys.argv) > 1 and sys.argv[1] in {"controller", "storage", "info"}:
        sub = sys.argv[1]
        rest = sys.argv[2:]
        sys.exit({"controller": controller_main, "storage": storage_main, "info": info_main}[sub](rest))
    print("Usage: python -m tq_rayless.cli {controller|storage|info} [args...]", file=sys.stderr)
    sys.exit(2)
