"""Training-only benchmark: feed the first N trajectories to a multi-GPU TitanTrainer.

No inference engine, no TransferQueue. Every rank builds the same
``TitanTrainer`` (any FSDP / HSDP / TP / CP layout), reads the first ``N``
records of a ``trajectories.jsonl``, tokenizes them with the model's chat
template (or reuses ``tokens/logprobs/masks`` when the log was written with
``verbose_trajectory_log=True``), plans the batch and runs ``--steps`` train
steps over it. Wall time per step, tokens/s and peak memory are printed on
rank 0 (and optionally dumped as JSON).

Examples::

    # 8-card FSDP, padded layout (the default), first 512 samples, 3 timed steps
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 \\
        scripts/bench_train_only.py --model-path /path/to/MiniCPM5-2B \\
        --model-name minicpm5 --model-flavor 2B --seq-len 131072 \\
        --trajectories .xrl_runtime/<run>/trajectories.jsonl --num-samples 512 \\
        --dp-replicate 2 --cp 4 --ac-mode full --compile --steps 3

    # same data, packed varlen layout with a 32k-token micro budget, no CP
    ... --layout packed --max-tokens-per-micro 32768 --cp 1

    # skip the HF weight load (random init) when only the timing matters
    ... --random-init

    # baseline: the pre-dynamic-batching behaviour (every micro-batch is a
    # fixed [micro_batch_size, seq_len] block, DP split by sample count)
    ... --plan legacy --layout padded --micro-batch-size 1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def _read_records(path: str, n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= n:
                break
    if len(out) < n:
        raise ValueError(f"{path} holds only {len(out)} records, {n} requested")
    return out


class _Tokenizer:
    """Rebuild ``tokens / mask_assistant`` from a message list.

    Non-assistant turns are tokenized through the chat template exactly like
    ``SampleBuilder.append_text`` does (template diff with the generation
    prompt); assistant turns are ``encode(content) + [eos]`` — what the
    inference engine actually produced, without a trailing generation prompt.
    """

    def __init__(self, model_path: str) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.eos = self.tok.eos_token_id

    def _template(self, messages: list[dict[str, str]]) -> list[int]:
        out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        # Newer transformers return a BatchEncoding (a UserDict, not a dict).
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def __call__(self, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
        tokens: list[int] = []
        masks: list[int] = []
        seen: list[dict[str, str]] = []
        for msg in messages:
            seen.append(msg)
            if msg["role"] == "assistant":
                ids = self.tok.encode(msg["content"], add_special_tokens=False)
                if self.eos is not None:
                    ids = ids + [self.eos]
                masks.extend([1] * len(ids))
            else:
                # Tokens the template adds beyond the previous turns (the
                # first turn is taken whole), mirroring SampleBuilder.append_text.
                full = self._template(seen)
                prev = self._template(seen[:-1]) if len(seen) > 1 else []
                ids = full[len(prev):]
                masks.extend([0] * len(ids))
            tokens.extend(ids)
        return tokens, masks


def _build_samples(records: list[dict[str, Any]], model_path: str, *, logprob_fill: float):
    import torch
    from tensordict import TensorDict

    tokenizer: _Tokenizer | None = None
    samples = []
    for rec in records:
        if "tokens" in rec and "masks" in rec:
            tokens = list(rec["tokens"])
            masks = list(rec["masks"])
            logprobs = list(rec.get("logprobs") or [logprob_fill] * len(tokens))
        else:
            if tokenizer is None:
                tokenizer = _Tokenizer(model_path)
            tokens, masks = tokenizer(rec["trajectory"])
            logprobs = [logprob_fill] * len(tokens)
        samples.append(
            TensorDict(
                {
                    "tokens": torch.tensor(tokens, dtype=torch.long),
                    "logprobs": torch.tensor(logprobs, dtype=torch.float32),
                    "mask_assistant": torch.tensor(masks, dtype=torch.float32),
                    "advantage": torch.tensor(float(rec.get("advantage") or 0.0)),
                    "weight_version": torch.tensor(int(rec.get("weight_version") or 0)),
                },
                batch_size=[],
            )
        )
    return samples


def _length_stats(samples, seq_len: int) -> dict[str, float]:
    lens = [min(int(s["tokens"].numel()), seq_len) for s in samples]
    resp = [int((s["mask_assistant"][:seq_len] > 0).sum()) for s in samples]
    return {
        "n": len(lens),
        "tokens_total": float(sum(lens)),
        "tokens_min": float(min(lens)),
        "tokens_mean": float(sum(lens) / len(lens)),
        "tokens_max": float(max(lens)),
        "response_tokens_total": float(sum(resp)),
        "truncated": float(sum(int(s["tokens"].numel()) > seq_len for s in samples)),
    }


def _legacy_plan(
    samples,
    dp_size: int,
    *,
    seq_len: int,
    mini_batch_size: int,
    micro_batch_size: int,
):
    """Rebuild the pre-dynamic-batching schedule as a :class:`Plan`.

    Before dynamic batching the trainer padded *every* sample to ``seq_len``
    (``pack_batch``), split the batch across DP ranks by contiguous sample
    slices (``split_batch_to_local``) and ran ``mini_batch_size //
    micro_batch_size`` equal micro-batches per optimizer step. The current
    ``build_micro_batch`` takes the padded row length from
    ``MicroPlan.seq_len``, so pinning it to ``seq_len`` for every micro
    reproduces exactly the old forward shapes ``[micro_batch_size, seq_len]``
    without touching the trainer.

    Loss normalisation differs from the old code (global counts instead of a
    per-rank mean), which is irrelevant for timing.
    """
    from meshy.backend.titan.plan import MicroPlan, MiniPlan, Plan
    from meshy.backend.titan.trainer import TitanTrainer

    lengths, loss_tokens = TitanTrainer._sample_lengths(samples, seq_len)
    n = len(samples)
    n_local = n // dp_size
    mini_batch_size = max(micro_batch_size, mini_batch_size)
    local_indices = tuple(
        tuple(range(r * n_local, (r + 1) * n_local)) for r in range(dp_size)
    )
    mini_bounds = [
        (lo, min(lo + mini_batch_size, n_local))
        for lo in range(0, n_local, mini_batch_size)
    ]

    per_rank = []
    for r in range(dp_size):
        minis = []
        for lo, hi in mini_bounds:
            micros = []
            for s in range(lo, hi, micro_batch_size):
                e = min(s + micro_batch_size, hi)
                idx = tuple(range(s, e))
                doc_lens = tuple(min(lengths[local_indices[r][p]], seq_len) for p in idx)
                micros.append(MicroPlan(idx, seq_len, doc_lens))
            n_docs_global = (hi - lo) * dp_size
            n_tokens_global = 0
            tokens_per_rank = []
            for q in range(dp_size):
                g = local_indices[q][lo:hi]
                n_tokens_global += sum(loss_tokens[i] for i in g)
                tokens_per_rank.append(sum(min(lengths[i], seq_len) for i in g))
            minis.append(
                MiniPlan(
                    micros=tuple(micros),
                    n_docs_global=n_docs_global,
                    n_tokens_global=n_tokens_global,
                    tokens_per_rank=tuple(tokens_per_rank),
                )
            )
        per_rank.append(tuple(minis))
    return Plan(per_rank=tuple(per_rank), local_indices=local_indices)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="HF model dir (weights + tokenizer)")
    p.add_argument("--model-name", default="minicpm5")
    p.add_argument("--model-flavor", default="2B")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--num-samples", "-n", type=int, required=True)
    p.add_argument("--seq-len", type=int, default=131072)
    p.add_argument("--steps", type=int, default=1, help="timed train steps over the same batch")
    p.add_argument("--warmup-steps", type=int, default=0, help="untimed steps first (compile warm-up)")
    # parallelism
    p.add_argument("--dp-shard", type=int, default=-1)
    p.add_argument("--dp-replicate", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    # batching
    p.add_argument("--layout", choices=["padded", "packed"], default="padded")
    p.add_argument(
        "--plan", choices=["dynamic", "legacy"], default="dynamic",
        help="'dynamic' = trainer.plan_batch (token-balanced DP split, dynamic "
             "micro shapes); 'legacy' = pre-dynamic-batching baseline (every "
             "micro is [micro_batch_size, seq_len], contiguous DP split). "
             "'legacy' requires --layout padded.",
    )
    p.add_argument("--mini-batch-size", type=int, default=4)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--max-tokens-per-micro", type=int, default=None)
    p.add_argument("--seq-bucket", type=int, default=2048)
    p.add_argument("--old-logprobs-source", choices=["rollout", "train"], default="rollout")
    p.add_argument("--per-token-loss", action="store_true")
    p.add_argument("--logprob-chunk-size", type=int, default=2048)
    # model / memory
    p.add_argument("--ac-mode", choices=["selective", "full", "memory_budget", "none"], default="full")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--random-init", action="store_true", help="skip the HF weight load")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--dump-folder", default="./outputs/bench_train_only")
    p.add_argument("--output", default=None, help="write the rank-0 summary JSON here")
    return p.parse_args()


def main() -> int:
    args = _parse()
    t_start = time.perf_counter()

    import torch
    import torch.distributed as dist

    from meshy.backend.titan.config import build_forge_config
    from meshy.backend.titan.trainer import TitanTrainer
    from meshy.config import TrainerConfig
    from meshy.utils.model import resolve_model_path

    model_path = resolve_model_path(args.model_path)
    attn_backend = "varlen" if args.layout == "packed" else "sdpa"
    if args.plan == "legacy" and args.layout != "padded":
        raise SystemExit("--plan legacy only makes sense with --layout padded")

    # ---- trainer ---------------------------------------------------------
    cfg = TrainerConfig(
        model_name=args.model_name,
        model_flavor=args.model_flavor,
        seq_len=args.seq_len,
        steps=max(1, args.steps + args.warmup_steps),
        lr=args.lr,
        dtype="bfloat16",
        compile_model=args.compile,
        activation_checkpoint_mode=args.ac_mode,
        dp_shard_degree=args.dp_shard,
        dp_replicate_degree=args.dp_replicate,
        tp_degree=args.tp,
        cp_degree=args.cp,
        enable_checkpoint=False,
        dump_folder=args.dump_folder,
        attn_backend=attn_backend,
    )
    forge = build_forge_config(cfg, hf_model_path=None if args.random_init else model_path)
    t0 = time.perf_counter()
    trainer = TitanTrainer(
        forge,
        mini_batch_size=args.mini_batch_size,
        micro_batch_size=args.micro_batch_size,
        batch_layout=args.layout,
        max_tokens_per_micro=args.max_tokens_per_micro,
        seq_bucket=args.seq_bucket,
        old_logprobs_source=args.old_logprobs_source,
        calculate_per_token_loss=args.per_token_loss,
        logprob_chunk_size=args.logprob_chunk_size,
        timer_enabled=True,
    )
    t_build = time.perf_counter() - t0
    rank = dist.get_rank()
    world = dist.get_world_size()
    dp_size = trainer.dp_degree
    device = trainer.device

    def log(msg: str) -> None:
        if rank == 0:
            print(f"[bench] {msg}", flush=True)

    log(f"trainer built in {t_build:.1f}s: world={world} dp={dp_size} tp={trainer.parallel_dims.tp} "
        f"cp={trainer.parallel_dims.cp} layout={args.layout} plan={args.plan} attn={attn_backend} "
        f"mini={args.mini_batch_size} micro={args.micro_batch_size} "
        f"max_tokens_per_micro={args.max_tokens_per_micro} compile={args.compile} ac={args.ac_mode}")

    # ---- data (every rank loads the same records; the plan picks its share)
    n = args.num_samples
    if n % dp_size != 0:
        raise SystemExit(f"--num-samples {n} must be divisible by the DP size {dp_size}")
    t0 = time.perf_counter()
    records = _read_records(args.trajectories, n)
    samples = _build_samples(records, model_path, logprob_fill=0.0)
    t_data = time.perf_counter() - t0
    stats = _length_stats(samples, args.seq_len)
    log(f"data ready in {t_data:.1f}s: {int(stats['n'])} samples, tokens "
        f"min/mean/max = {stats['tokens_min']:.0f}/{stats['tokens_mean']:.0f}/{stats['tokens_max']:.0f}, "
        f"total {stats['tokens_total']:.0f} (response {stats['response_tokens_total']:.0f}), "
        f"truncated to seq_len: {int(stats['truncated'])}")

    if args.plan == "legacy":
        plan = _legacy_plan(
            samples, dp_size, seq_len=args.seq_len,
            mini_batch_size=args.mini_batch_size, micro_batch_size=args.micro_batch_size,
        )
    else:
        plan = trainer.plan_batch(samples, dp_size)
    local = plan.local_samples(samples, trainer.dp_rank)
    local_plan = plan.per_rank[trainer.dp_rank]
    from meshy.backend.titan.plan import plan_stats

    ps = plan_stats(local_plan, args.layout)
    log(f"plan (rank 0 view): minis={len(local_plan)} micros={int(ps['n_micro'])} "
        f"tokens/micro={ps['tokens_per_micro']:.0f} padding_ratio={ps['padding_ratio']:.3f} "
        f"dp_token_imbalance={ps['dp_token_imbalance']:.3f}")

    # ---- train -----------------------------------------------------------
    def one_step() -> tuple[float, dict[str, float]]:
        torch.cuda.synchronize(device)
        dist.barrier()
        t = time.perf_counter()
        metrics = trainer.train_step(local, plan=local_plan)
        torch.cuda.synchronize(device)
        dist.barrier()
        return time.perf_counter() - t, metrics

    torch.cuda.reset_peak_memory_stats(device)
    for i in range(args.warmup_steps):
        dt, _ = one_step()
        log(f"warmup step {i}: {dt:.2f}s")

    step_times: list[float] = []
    last_metrics: dict[str, float] = {}
    for i in range(args.steps):
        dt, last_metrics = one_step()
        step_times.append(dt)
        log(f"step {i}: {dt:.2f}s  loss={last_metrics.get('pg_loss', float('nan')):.4f} "
            f"grad_norm={last_metrics.get('grad_norm', float('nan')):.3f} "
            f"fwd={last_metrics.get('time/train/forward', 0):.2f}s "
            f"bwd={last_metrics.get('time/train/backward', 0):.2f}s "
            f"old_lp={last_metrics.get('time/train/old_logprobs', 0):.2f}s "
            f"optim={last_metrics.get('time/train/optim_step', 0):.2f}s")

    peak = torch.cuda.max_memory_allocated(device) / 2**30
    peak_all = torch.tensor([peak], device=device)
    dist.all_reduce(peak_all, op=dist.ReduceOp.MAX)

    mean_step = statistics.fmean(step_times)
    summary = {
        "config": {
            "model": f"{args.model_name}/{args.model_flavor}", "seq_len": args.seq_len,
            "world": world, "dp": dp_size, "tp": trainer.parallel_dims.tp, "cp": trainer.parallel_dims.cp,
            "layout": args.layout, "plan": args.plan, "attn_backend": attn_backend,
            "mini_batch_size": args.mini_batch_size, "micro_batch_size": args.micro_batch_size,
            "max_tokens_per_micro": args.max_tokens_per_micro, "seq_bucket": args.seq_bucket,
            "compile": args.compile, "ac_mode": args.ac_mode,
            "old_logprobs_source": args.old_logprobs_source, "num_samples": n,
        },
        "data": stats,
        "plan_rank0": ps,
        "setup": {"trainer_build_s": t_build, "data_load_s": t_data,
                  "total_before_train_s": time.perf_counter() - t_start},
        "steps": step_times,
        "step_time_mean_s": mean_step,
        "step_time_min_s": min(step_times),
        "tokens_per_s_total": stats["tokens_total"] / mean_step,
        "tokens_per_s_per_gpu": stats["tokens_total"] / mean_step / world,
        "peak_mem_gib_max_rank": float(peak_all.item()),
        "last_metrics": {k: float(v) for k, v in last_metrics.items()
                         if isinstance(v, (int, float))},
    }
    if rank == 0:
        print("\n===== bench_train_only summary =====")
        print(f"samples={n} tokens={stats['tokens_total']:.0f} world={world} "
              f"(dp={dp_size} tp={trainer.parallel_dims.tp} cp={trainer.parallel_dims.cp}) "
              f"layout={args.layout} plan={args.plan}")
        print(f"step time: mean {mean_step:.2f}s  min {min(step_times):.2f}s  over {len(step_times)} step(s)")
        print(f"throughput: {summary['tokens_per_s_total']:.0f} tok/s total, "
              f"{summary['tokens_per_s_per_gpu']:.0f} tok/s/GPU")
        print(f"peak memory (max over ranks): {summary['peak_mem_gib_max_rank']:.1f} GiB")
        print(f"padding_ratio={ps['padding_ratio']:.3f} micros/rank={int(ps['n_micro'])} "
              f"dp_token_imbalance={ps['dp_token_imbalance']:.3f}")
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(summary, indent=1))
            print(f"summary written to {args.output}")

    dist.barrier()
    trainer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
