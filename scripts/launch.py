"""Launcher wrapper: run a recipe under a single ``torchrun``.

Usage::

    python scripts/launch.py --recipe recipe.grpo_gsm8k

The launcher imports the recipe's ``SERVICE_GROUPS`` (cheap: dataclasses only),
computes the total card count, and launches **one** ``torchrun`` with one
ignitor process per card. Colocate (inference + trainer sharing cards) needs no
second torchrun: each shared card's ignitor ignites the inference replica first
(which frees its GPU memory on ready) then the colocate trainer, whose
``wait_until`` gate blocks on the inference readiness markers.

For multi-node, run this on every node with matching ``--nnodes`` /
``--node-rank`` / ``--master-addr`` and an explicit ``--nproc-per-node``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import subprocess
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Meshy service launcher")
    ap.add_argument("--recipe", default="recipe.grpo_gsm8k", help="recipe module path")
    ap.add_argument("--nnodes", type=int, default=1)
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--master-addr", default="127.0.0.1")
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="cards per node (defaults to total cards; required for multi-node)",
    )
    ap.add_argument("--runtime-dir", default=None, help="shared runtime dir (default: timestamped)")
    ap.add_argument("--checkpoint-dir", default=None, help="checkpoint output directory (default: <runtime>/weights)")
    ap.add_argument("--log-dir", default=None, help="service and engine log directory (default: <runtime>/logs)")
    ap.add_argument("--tensorboard-dir", default=None, help="TensorBoard event directory (default: <runtime>/tensorboard)")
    ap.add_argument("--trajectory-dir", default=None, help="trajectory JSONL directory (default: <runtime>)")
    return ap


def _torchrun_cmd(args, nproc: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(args.nnodes),
        "--node-rank",
        str(args.node_rank),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "--nproc-per-node",
        str(nproc),
        "-m",
        args.recipe,
    ]


def main() -> None:
    args = _build_parser().parse_args()

    # Make the recipe importable from the repo root both here (to compute the
    # card count) and in the torchrun children (via PYTHONPATH), since an
    # editable install only exposes the ``meshy`` package, not sibling ``recipe``.
    repo_root = os.getcwd()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Resolve the runtime dir *before* importing the recipe: a recipe may derive
    # paths from XRL_RUNTIME_DIR at import time, and the torchrun children import
    # the same module with this variable already set. Deciding it afterwards
    # would give the launcher and the ignitors two different answers.
    runtime_dir = args.runtime_dir or os.environ.get("XRL_RUNTIME_DIR")
    if not runtime_dir:
        runtime_dir = os.path.abspath(os.path.join(".xrl_runtime", time.strftime("%Y%m%d-%H%M%S")))
    os.environ["XRL_RUNTIME_DIR"] = runtime_dir
    os.makedirs(runtime_dir, exist_ok=True)
    for arg_name, env_name in (
        ("checkpoint_dir", "XRL_CHECKPOINT_DIR"),
        ("log_dir", "XRL_LOG_DIR"),
        ("tensorboard_dir", "XRL_TENSORBOARD_DIR"),
        ("trajectory_dir", "XRL_TRAJECTORY_DIR"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            os.environ[env_name] = value

    recipe = importlib.import_module(args.recipe)
    service_groups = recipe.SERVICE_GROUPS
    total_cards = sum(sg.n_gpus for sg in service_groups)
    has_colocate = any(sg.colocate_with is not None for sg in service_groups)
    nproc = args.nproc_per_node or total_cards
    if nproc <= 0:
        raise SystemExit("no GPU cards required by recipe; nothing to launch")

    # Bootstrap store: the run's KV plane for startup synchronization
    # (readiness markers, TQ endpoint discovery). Node 0 hosts; every child on
    # every node connects through XRL_BOOTSTRAP_ADDR, so multi-node runs need
    # no shared filesystem for synchronization. A pre-set XRL_BOOTSTRAP_ADDR
    # means an external process hosts the store -- then nobody here hosts.
    from meshy.service import bootstrap

    if "XRL_BOOTSTRAP_ADDR" not in os.environ:
        port = int(os.environ.get("XRL_BOOTSTRAP_PORT", args.master_port + 1))
        os.environ["XRL_BOOTSTRAP_ADDR"] = f"{args.master_addr}:{port}"
        if args.node_rank == 0:
            bootstrap.host_store(args.master_addr, port)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    print(f"[launch] recipe={args.recipe} cards={total_cards} nproc-per-node={nproc} "
          f"colocate={has_colocate} runtime={runtime_dir}", flush=True)

    # TransferQueue is mandatory infrastructure: the data plane (samples) and
    # the control plane (gen gates) both ride on it, so it must be up before
    # any Service starts (they connect during ignition). Node 0 owns the
    # cluster; every node discovers the endpoints through the bootstrap store.
    # A recipe may export TRANSFER_QUEUE to override tuning knobs; otherwise
    # the spec is derived from the typed service configs.
    tq_cluster = None
    if args.node_rank == 0:
        from meshy.transferqueue.launch import TransferQueueCluster
        from meshy.transferqueue.spec import derive_tq_spec

        tq_spec = getattr(recipe, "TRANSFER_QUEUE", None) or derive_tq_spec(
            service_groups, runtime_dir
        )
        tq_cluster = TransferQueueCluster(
            tq_spec["endpoints_file"],
            num_storage_units=int(tq_spec.get("num_storage_units", 1)),
            storage_unit_size=int(tq_spec.get("storage_unit_size", 100_000)),
            pre_alloc_sample_num=int(tq_spec["pre_alloc_sample_num"]),
            # Keep all launcher-owned process logs together, including the
            # TransferQueue controller and storage workers.
            log_dir=os.environ.get("XRL_LOG_DIR") or os.path.join(runtime_dir, "logs"),
        ).start()

    proc = subprocess.Popen(_torchrun_cmd(args, nproc), env=env)

    def _shutdown(*_):
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        proc.wait()
    finally:
        _shutdown()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        if tq_cluster is not None:
            tq_cluster.stop()

    sys.exit(proc.returncode or 0)


if __name__ == "__main__":
    main()
