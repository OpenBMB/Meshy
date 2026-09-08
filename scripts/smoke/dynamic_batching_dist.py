"""Distributed check for dynamic batching: DP / CP runs must match one rank.

Run the single-rank reference first, then any parallel layout; the parallel
run loads the reference metrics and asserts loss / grad_norm agree. Weights
are set deterministically from the parameter names so every run trains the
same model regardless of how it is sharded.

    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 scripts/smoke/dynamic_batching_dist.py \
        --layout packed --ref /tmp/dyn_ref_packed.json --write
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/smoke/dynamic_batching_dist.py \
        --layout packed --ref /tmp/dyn_ref_packed.json --dp-shard 2
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/smoke/dynamic_batching_dist.py \
        --layout padded --ref /tmp/dyn_ref_padded.json --cp 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEQ_LEN = 512
LENGTHS = (300, 120, 50, 200, 33, 260, 410, 77)


def _samples(lengths=LENGTHS, seed: int = 0):
    import torch
    from tensordict import TensorDict

    g = torch.Generator().manual_seed(seed)
    out = []
    for L in lengths:
        mask = torch.ones(L)
        mask[: L // 3] = 0
        out.append(TensorDict({
            "tokens": torch.randint(1, 2000, (L,), generator=g),
            "logprobs": -torch.rand(L, generator=g) * 2,
            "mask_assistant": mask,
            "advantage": torch.randn((), generator=g),
            "weight_version": torch.tensor(0),
        }, batch_size=[]))
    return out


def _deterministic_weights(trainer):
    """Fill every parameter from a hash of its name, without any collective.

    Every rank builds the same full tensor (DTensor ``.shape`` is global) and
    keeps its own shard via ``distribute_tensor(src_data_rank=None)``.
    """
    import hashlib

    import torch
    from torch.distributed.tensor import DTensor, distribute_tensor

    with torch.no_grad():
        for name, param in trainer.model_parts[0].named_parameters():
            seed = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            std = 1.0 if "norm" in name else 0.02
            full = torch.randn(param.shape, generator=g, dtype=torch.float32) * std
            if "norm" in name:
                full = 1.0 + 0.1 * full
            full = full.to(param.device)
            if isinstance(param, DTensor):
                full = distribute_tensor(
                    full, param.device_mesh, param.placements, src_data_rank=None
                )
            param.copy_(full)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=["padded", "packed"], default="padded")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--dp-shard", type=int, default=-1)
    parser.add_argument("--dp-replicate", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--per-token", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--lengths", default=None,
                        help="comma-separated sample lengths (default: built-in set)")
    args = parser.parse_args()
    lengths = tuple(int(x) for x in args.lengths.split(",")) if args.lengths else LENGTHS

    import torch
    import torch.distributed as dist

    from meshy.backend.titan.config import build_forge_config
    from meshy.backend.titan.parallel import split_batch_to_local
    from meshy.backend.titan.trainer import TitanTrainer
    from meshy.config import TrainerConfig

    cfg = TrainerConfig(
        model_name="qwen3", model_flavor="debugmodel", seq_len=SEQ_LEN,
        attn_backend="varlen" if args.layout == "packed" else "sdpa",
        dp_shard_degree=args.dp_shard, dp_replicate_degree=args.dp_replicate,
        cp_degree=args.cp, tp_degree=args.tp,
        enable_checkpoint=False, compile_model=args.compile,
        dump_folder="/tmp/xrl_dyn_batch_dist", lr=0.0, max_norm=1e9,
    )
    # ``train`` behaviour policy: ratio == 1 at step 0, so nothing is clipped
    # and the gradient is a plain -adv * grad(log p) -- a meaningful signal
    # for comparing layouts / parallelisms (random rollout logprobs would
    # clip every token).
    params = dict(mini_batch_size=8, seq_bucket=64, timer_enabled=False,
                  calculate_per_token_loss=args.per_token,
                  old_logprobs_source="train")
    if args.layout == "packed":
        params["max_tokens_per_micro"] = 640
    else:
        params["micro_batch_size"] = 3
    trainer = TitanTrainer(build_forge_config(cfg), batch_layout=args.layout, **params)
    _deterministic_weights(trainer)

    world = dist.get_world_size()
    rank = dist.get_rank()
    samples = _samples(lengths)
    if world == 1:
        metrics = trainer.train_step(samples)
    else:
        local, plan = split_batch_to_local(
            samples if rank == 0 else None, trainer.parallel_dims,
            planner=trainer.plan_batch,
        )
        metrics = trainer.train_step(local, plan=plan)

    keys = ["pg_loss", "grad_norm", "ratio_mean", "clip_frac", "train/n_micro",
            "train/padding_ratio", "train/dp_token_imbalance"]
    got = {k: float(metrics[k]) for k in keys}
    # Metrics must agree on every rank (they are globally reduced).
    gathered = [None] * world
    dist.all_gather_object(gathered, got)
    if rank == 0:
        print(json.dumps({"world": world, "cp": args.cp, "layout": args.layout, "metrics": got}, indent=1))
        for r, other in enumerate(gathered):
            for k in ("pg_loss", "grad_norm", "ratio_mean", "clip_frac"):
                assert abs(other[k] - got[k]) < 1e-6, f"rank {r} disagrees on {k}: {other[k]} vs {got[k]}"
        if args.write:
            Path(args.ref).write_text(json.dumps(got))
            print("reference written")
        else:
            ref = json.loads(Path(args.ref).read_text())
            for k in ("pg_loss", "grad_norm", "ratio_mean"):
                tol = 2e-2 * max(1.0, abs(ref[k]))
                assert abs(ref[k] - got[k]) < tol, f"{k}: ref {ref[k]} vs {got[k]}"
            print("MATCH reference")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
