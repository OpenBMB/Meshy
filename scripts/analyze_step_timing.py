"""Analyze per-step / per-phase wall-clock timing from an Meshy run log.

The launcher writes a combined log (e.g. ``meshy.log``) that interleaves
messages from the AgentLoop worker, the Titan trainer and the SGLang inference
servers.  Depending on the recipe / agentloop mode, one of three drivers
produced the log; this script auto-detects which and attributes the wall clock
accordingly (force it with ``--mode``).

Relevant markers::

    meshy.service.training:_train_and_sync   Trainer ... step (v0 -> v1) metrics: ... time/train/...
    meshy.service.training:_train_and_sync   Trainer ... stream-train chunk (v0, no sync) metrics: ...
    meshy.service.training:_sync_inference   Trainer ... synced weights v1 to N inference engine(s) in T.Ts
    meshy.backend.titan.trainer:save_hf_checkpoint  save_hf_checkpoint: done -> .../v1
    meshy.service.agentloop:_amain        AgentLoop: rollout for version 0+1 (...)   [sync driver]
    meshy.service.agentloop:_wait_version AgentLoop: version advanced 0 -> 1         [sync driver]

Three drivers, three ways to attribute the wall clock:

* **sync driver** (``_amain``): rollout is a discrete phase that starts with an
  explicit "rollout for version X+1" line, so each step splits cleanly into
  rollout / train / sync / version-handoff.

* **async driver** (``_amain_async``): rollout runs continuously and overlaps
  with training, but the trainer still performs one ``train_step`` per version
  (one "step (vX -> vX+1)" line per step).  The gap between two consecutive
  training steps is the step length::

      total(step N)     = train_done[N] - train_done[N-1]
      train(step N)     = sum(time/train/*) of step N
      sync(step N)      = weight-sync seconds that fall inside that gap
      inference(step N) = total - train - sync

* **stream driver** (``_amain_async`` + ``stream_minibatch``): the fully-async
  recipe (e.g. ``recipe.justrl_fully_async``).  The trainer trains on many
  ``mini_batch_size * dp_size`` **chunks** as they stream in -- each logs a
  "stream-train chunk (vX, no sync)" line -- and only the chunk that closes a
  full ``batch_size`` window logs "step (vX -> vX+1)" and triggers the
  checkpoint dump + weight sync + version bump.  A step (version window
  ``v(N-1) -> vN``) is therefore attributed as::

      total(step N)   = sync_done[N] - sync_done[N-1]   (step 1: from 1st chunk)
      train(step N)   = sum of every chunk's time/train/* in the window
                        (all "no sync" chunks + the closing "step" chunk)
      ckpt(step N)    = save_hf_checkpoint done_ts - closing step_ts
      sync(step N)    = reported "synced weights vN in T.Ts"
      rollout(step N) = total - train - ckpt - sync   (generation-bound idle
                        time the trainer spent waiting for the next chunk)

Usage::

    python scripts/analyze_step_timing.py
    python scripts/analyze_step_timing.py .xrl_runtime/20260710-035025/meshy.log
    python scripts/analyze_step_timing.py --mode stream --csv step_timing.csv run.log
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# xrl / loguru lines:  "2026-07-09 05:29:10.357 | INFO     | module:func:line - msg"
_XRL_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)"

_ROLLOUT_RE = re.compile(
    _XRL_TS + r".*AgentLoop: rollout for version (\d+)\+(\d+) "
    r"\((\d+) prompts x (\d+) group\)"
)
# closing / synchronous train step:  "step (v0 -> v1) metrics: ..."
_TRAIN_RE = re.compile(_XRL_TS + r".*step \(v(\d+) -> v(\d+)\) metrics: (.*)$")
# fully-async streamed mini-batch chunk:  "stream-train chunk (v0, no sync) metrics: ..."
_CHUNK_RE = re.compile(_XRL_TS + r".*stream-train chunk \(v(\d+), no sync\) metrics: (.*)$")
_SYNC_RE = re.compile(
    _XRL_TS + r".*synced weights v(\d+) to (\d+) inference engine\(s\) in ([\d.]+)s"
)
_CKPT_DONE_RE = re.compile(_XRL_TS + r".*save_hf_checkpoint: done.*?/v(\d+)\b")
_VERSION_RE = re.compile(_XRL_TS + r".*version advanced (\d+) -> (\d+)")
# key=value pairs inside the metrics blob, e.g. "time/train/forward=108.034"
_KV_RE = re.compile(r"([A-Za-z0-9_/]+)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")

_TS_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def _parse_ts(text: str) -> datetime:
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognized timestamp: {text!r}")


def _train_compute(metrics: dict[str, float]) -> float:
    return sum(v for k, v in metrics.items() if k.startswith("time/train/"))


def _train_subphases(metrics: dict[str, float]) -> dict[str, float]:
    return {
        k.split("time/train/")[1]: v
        for k, v in metrics.items()
        if k.startswith("time/train/")
    }


@dataclass
class Step:
    """One RL step: trains v(from_v) -> v(to_v)."""

    from_v: int
    to_v: int
    rollout_ts: datetime | None = None       # rollout for version from_v+1 started
    train_ts: datetime | None = None         # closing train step logged (train done)
    sync_ts: datetime | None = None          # this step's weights synced
    sync_secs: float | None = None           # this step's reported sync seconds
    ckpt_done_ts: datetime | None = None      # save_hf_checkpoint done for to_v
    version_ts: datetime | None = None       # AgentLoop saw version advance
    next_rollout_ts: datetime | None = None  # rollout of the following step
    prompts: int | None = None
    group: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    # streamed mini-batch chunks that belong to this version window (stream mode)
    chunk_metrics: list[dict[str, float]] = field(default_factory=list)
    chunk_ts: list[datetime] = field(default_factory=list)

    # filled in during post-processing
    window_start_ts: datetime | None = None
    total_secs: float | None = None
    inference_secs: float | None = None      # inference / rollout (idle) seconds
    train_secs: float | None = None          # train GPU compute attributed to step
    ckpt_secs: float | None = None           # checkpoint dump seconds (stream mode)
    sync_in_step_secs: float | None = None   # weight sync attributed to this step
    subphases: dict[str, float] = field(default_factory=dict)  # aggregated train sub-phases

    @property
    def train_compute(self) -> float:
        """Closing-step train compute (single train_step)."""
        return _train_compute(self.metrics)

    @property
    def train_subphases(self) -> dict[str, float]:
        return _train_subphases(self.metrics)

    @property
    def n_chunks(self) -> int:
        """Number of train_step invocations in this window (chunks + closing)."""
        return len(self.chunk_metrics) + (1 if self.metrics else 0)


@dataclass
class ParsedLog:
    steps: list[Step]
    syncs: dict[int, tuple[datetime, float]]  # version -> (ts, seconds)
    has_rollout_markers: bool
    has_stream_chunks: bool


def parse_log(path: Path) -> ParsedLog:
    steps: dict[int, Step] = {}
    syncs: dict[int, tuple[datetime, float]] = {}
    rollout_starts: list[tuple[int, datetime]] = []
    has_rollout_markers = False
    has_stream_chunks = False

    def step_for(to_v: int, from_v: int) -> Step:
        s = steps.get(to_v)
        if s is None:
            s = Step(from_v=from_v, to_v=to_v)
            steps[to_v] = s
        return s

    with path.open("r", errors="replace") as fh:
        for line in fh:
            if "rollout for version" in line:
                m = _ROLLOUT_RE.search(line)
                if m:
                    has_rollout_markers = True
                    ts = _parse_ts(m.group(1))
                    from_v = int(m.group(2))
                    to_v = from_v + 1
                    s = step_for(to_v, from_v)
                    s.rollout_ts = ts
                    s.prompts = int(m.group(4))
                    s.group = int(m.group(5))
                    rollout_starts.append((to_v, ts))
                continue
            if "stream-train chunk (v" in line:
                m = _CHUNK_RE.search(line)
                if m:
                    has_stream_chunks = True
                    from_v = int(m.group(2))
                    to_v = from_v + 1
                    s = step_for(to_v, from_v)
                    s.chunk_metrics.append(
                        {k: float(v) for k, v in _KV_RE.findall(m.group(3))}
                    )
                    s.chunk_ts.append(_parse_ts(m.group(1)))
                continue
            if "metrics:" in line and "step (v" in line:
                m = _TRAIN_RE.search(line)
                if m:
                    from_v, to_v = int(m.group(2)), int(m.group(3))
                    s = step_for(to_v, from_v)
                    s.train_ts = _parse_ts(m.group(1))
                    s.metrics = {k: float(v) for k, v in _KV_RE.findall(m.group(4))}
                continue
            if "save_hf_checkpoint: done" in line:
                m = _CKPT_DONE_RE.search(line)
                if m:
                    to_v = int(m.group(2))
                    if to_v >= 1:
                        step_for(to_v, to_v - 1).ckpt_done_ts = _parse_ts(m.group(1))
                continue
            if "synced weights v" in line:
                m = _SYNC_RE.search(line)
                if m:
                    ts = _parse_ts(m.group(1))
                    v = int(m.group(2))
                    secs = float(m.group(4))
                    syncs[v] = (ts, secs)
                    if v >= 1:  # v0 is the startup sync, not a train step
                        s = step_for(v, v - 1)
                        s.sync_ts = ts
                        s.sync_secs = secs
                continue
            if "version advanced" in line:
                m = _VERSION_RE.search(line)
                if m:
                    to_v = int(m.group(3))
                    step_for(to_v, int(m.group(2))).version_ts = _parse_ts(m.group(1))
                continue

    for to_v, ts in rollout_starts:
        prev = steps.get(to_v - 1)
        if prev is not None:
            prev.next_rollout_ts = ts

    ordered = [steps[v] for v in sorted(steps)]
    return ParsedLog(ordered, syncs, has_rollout_markers, has_stream_chunks)


def _compute_sync(steps: list[Step], syncs: dict[int, tuple[datetime, float]]) -> None:
    """Fill total/inference for the synchronous (discrete-rollout) driver."""
    for s in steps:
        if s.train_ts is None:
            continue
        s.train_secs = s.train_compute
        s.subphases = s.train_subphases
        s.sync_in_step_secs = s.sync_secs
        end = s.next_rollout_ts or s.version_ts or s.sync_ts
        if s.rollout_ts and end:
            s.total_secs = (end - s.rollout_ts).total_seconds()
        if s.rollout_ts and s.train_ts:
            e2e = (s.train_ts - s.rollout_ts).total_seconds()
            s.inference_secs = max(e2e - s.train_compute, 0.0)


def _compute_async(steps: list[Step], syncs: dict[int, tuple[datetime, float]]) -> None:
    """Fill total/inference for the async (continuous-rollout) driver.

    total(N)     = train_done[N] - train_done[N-1]   (step 1: from the v0 sync)
    sync(N)      = seconds of the weight sync that lands inside that interval,
                   i.e. the sync of version N-1 (v0 for step 1)
    inference(N) = total - train_compute(N) - sync(N)
    """
    trained = [s for s in steps if s.train_ts is not None]
    v0_sync = syncs.get(0)
    for i, s in enumerate(trained):
        s.train_secs = s.train_compute
        s.subphases = s.train_subphases
        if i == 0:
            prev_ts = v0_sync[0] if v0_sync else None
        else:
            prev_ts = trained[i - 1].train_ts
        # the weight sync at the start of this interval is the one for the
        # previous version (v0 for the first step)
        prev_sync = syncs.get(s.from_v)
        s.sync_in_step_secs = prev_sync[1] if prev_sync else None
        if prev_ts is not None and s.train_ts is not None:
            s.total_secs = (s.train_ts - prev_ts).total_seconds()
            sync_secs = s.sync_in_step_secs or 0.0
            s.inference_secs = max(s.total_secs - s.train_compute - sync_secs, 0.0)


def _compute_stream(steps: list[Step], syncs: dict[int, tuple[datetime, float]]) -> None:
    """Fill per-step timing for the fully-async streamed mini-batch driver.

    A step is the version window ``v(N-1) -> vN`` that ends with a weight sync.
    Wall-clock is fully partitioned between consecutive sync completions::

        total    = sync_done[N] - sync_done[N-1]   (step 1: from the 1st chunk)
        train    = sum of time/train/* over every chunk + the closing step
        ckpt     = checkpoint dump: ckpt_done_ts - closing step_ts
        sync     = reported "synced weights vN in T.Ts"
        rollout  = total - train - ckpt - sync   (generation-bound idle wait)
    """
    # only version windows that actually closed with a sync are complete steps
    windows = [s for s in steps if s.sync_ts is not None and s.train_ts is not None]
    for s in windows:
        # aggregate train compute + sub-phases over all chunks + closing step
        all_metrics = s.chunk_metrics + ([s.metrics] if s.metrics else [])
        s.train_secs = sum(_train_compute(m) for m in all_metrics)
        agg: dict[str, float] = {}
        for m in all_metrics:
            for k, v in _train_subphases(m).items():
                agg[k] = agg.get(k, 0.0) + v
        s.subphases = agg

        # window start = previous version's sync completion; for the first
        # window fall back to the earliest chunk timestamp we saw.
        prev_sync = syncs.get(s.from_v)
        if prev_sync is not None:
            s.window_start_ts = prev_sync[0]
        elif s.chunk_ts:
            s.window_start_ts = min(s.chunk_ts)
        else:
            s.window_start_ts = s.train_ts

        s.sync_in_step_secs = s.sync_secs
        # checkpoint dump: from the closing train step to "save_hf done"
        if s.ckpt_done_ts is not None and s.train_ts is not None:
            s.ckpt_secs = max((s.ckpt_done_ts - s.train_ts).total_seconds(), 0.0)
        else:
            s.ckpt_secs = 0.0

        if s.window_start_ts is not None and s.sync_ts is not None:
            s.total_secs = (s.sync_ts - s.window_start_ts).total_seconds()
            rollout = s.total_secs - s.train_secs - s.ckpt_secs - (s.sync_secs or 0.0)
            s.inference_secs = max(rollout, 0.0)


def _fmt(sec: float | None) -> str:
    return "       -" if sec is None else f"{sec:8.1f}"


def _fmt_mmss(sec: float | None) -> str:
    if sec is None:
        return "   -"
    m, s = divmod(int(round(sec)), 60)
    return f"{m:d}m{s:02d}s"


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _completed(parsed: ParsedLog, mode: str) -> list[Step]:
    if mode == "stream":
        return [s for s in parsed.steps if s.total_secs is not None]
    return [s for s in parsed.steps if s.train_ts is not None]


def _print_subphase_breakdown(steps: list[Step]) -> None:
    subkeys: list[str] = []
    for s in steps:
        for k in s.subphases:
            if k not in subkeys:
                subkeys.append(k)
    if not subkeys:
        return
    print()
    print("=" * 96)
    print("Train compute sub-phase breakdown (seconds, summed over the step)")
    print("=" * 96)
    head = f"{'step':>4} " + " ".join(f"{k[:12]:>12}" for k in subkeys) + f"{'sum':>10}"
    print(head)
    print("-" * len(head))
    for s in steps:
        step_no = int(s.metrics.get("step", s.to_v))
        row = f"{step_no:>4} " + " ".join(
            f"{s.subphases.get(k, 0.0):12.2f}" for k in subkeys
        )
        print(row + f"{sum(s.subphases.values()):10.2f}")

    agg_sub: dict[str, float] = {}
    for s in steps:
        for k, v in s.subphases.items():
            agg_sub[k] = agg_sub.get(k, 0.0) + v
    n = len(steps)
    tot = sum(agg_sub.values()) or 1.0
    print()
    print("avg train sub-phase (seconds/step, share of train compute):")
    for k, v in sorted(agg_sub.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k:<16}: {v / n:8.2f}s  ({v / tot * 100:5.1f}%)")


def print_report(parsed: ParsedLog, mode: str) -> None:
    if mode == "stream":
        print_report_stream(parsed)
        return

    steps = _completed(parsed, mode)
    if not steps:
        print("No completed training steps found in the log.")
        return

    print(f"driver mode: {mode}")
    print("=" * 96)
    print("Per-step phase timing (seconds)")
    print("=" * 96)
    print(
        f"{'step':>4} {'v->v':>8} {'total':>8} {'inference':>10} "
        f"{'train':>8} {'sync':>7}  {'total(mmss)':>11}"
    )
    print("-" * 96)
    for s in steps:
        step_no = int(s.metrics.get("step", s.to_v))
        print(
            f"{step_no:>4} {f'v{s.from_v}->v{s.to_v}':>8} {_fmt(s.total_secs)} "
            f"{_fmt(s.inference_secs)} {_fmt(s.train_compute)} "
            f"{_fmt(s.sync_in_step_secs)}  {_fmt_mmss(s.total_secs):>11}"
        )

    _print_subphase_breakdown(steps)

    inf = [s.inference_secs for s in steps if s.inference_secs is not None]
    train = [s.train_compute for s in steps]
    sync = [s.sync_in_step_secs for s in steps if s.sync_in_step_secs is not None]
    totals = [s.total_secs for s in steps if s.total_secs is not None]

    print()
    print("=" * 96)
    print("Aggregate")
    print("=" * 96)
    print(f"completed steps        : {len(steps)}")
    if totals:
        print(f"avg step total         : {_avg(totals):8.1f}s ({_fmt_mmss(_avg(totals))})")
    print(f"avg inference/rollout  : {_avg(inf):8.1f}s")
    print(f"avg train compute      : {_avg(train):8.1f}s")
    print(f"avg weight sync        : {_avg(sync):8.1f}s")

    ai, at, asy = _avg(inf), _avg(train), _avg(sync)
    denom = ai + at + asy
    if denom > 0:
        print()
        print("phase share (inference + train + sync):")
        print(f"  inference/rollout : {ai / denom * 100:5.1f}%")
        print(f"  train compute     : {at / denom * 100:5.1f}%")
        print(f"  weight sync       : {asy / denom * 100:5.1f}%")


def print_report_stream(parsed: ParsedLog) -> None:
    steps = _completed(parsed, "stream")
    if not steps:
        print("No completed (synced) training steps found in the log.")
        return

    print("driver mode: stream (fully-async streamed mini-batch)")
    print("=" * 100)
    print("Per-step phase timing (seconds) -- total = rollout + train + ckpt + sync")
    print("=" * 100)
    print(
        f"{'step':>4} {'v->v':>8} {'total':>8} {'rollout':>9} {'train':>9} "
        f"{'ckpt':>6} {'sync':>7} {'chunks':>6}  {'total(mmss)':>11}"
    )
    print("-" * 100)
    for s in steps:
        step_no = int(s.metrics.get("step", s.to_v))
        print(
            f"{step_no:>4} {f'v{s.from_v}->v{s.to_v}':>8} {_fmt(s.total_secs)} "
            f"{_fmt(s.inference_secs)} {_fmt(s.train_secs)} "
            f"{(s.ckpt_secs or 0.0):6.1f} {_fmt(s.sync_in_step_secs)} "
            f"{s.n_chunks:>6}  {_fmt_mmss(s.total_secs):>11}"
        )

    _print_subphase_breakdown(steps)

    totals = [s.total_secs for s in steps if s.total_secs is not None]
    roll = [s.inference_secs for s in steps if s.inference_secs is not None]
    train = [s.train_secs for s in steps if s.train_secs is not None]
    ckpt = [s.ckpt_secs for s in steps if s.ckpt_secs is not None]
    sync = [s.sync_in_step_secs for s in steps if s.sync_in_step_secs is not None]
    chunks = [s.n_chunks for s in steps if s.n_chunks]

    print()
    print("=" * 100)
    print("Aggregate")
    print("=" * 100)
    print(f"completed steps        : {len(steps)}")
    if totals:
        print(f"avg step total         : {_avg(totals):8.1f}s ({_fmt_mmss(_avg(totals))})")
    print(f"avg rollout (idle wait): {_avg(roll):8.1f}s")
    print(f"avg train compute      : {_avg(train):8.1f}s")
    print(f"avg checkpoint dump    : {_avg(ckpt):8.1f}s")
    print(f"avg weight sync        : {_avg(sync):8.1f}s")
    if chunks:
        print(f"avg chunks/step        : {_avg([float(c) for c in chunks]):8.1f}")

    ar, at, ac, asy = _avg(roll), _avg(train), _avg(ckpt), _avg(sync)
    denom = ar + at + ac + asy
    if denom > 0:
        print()
        print("phase share of step wall-clock (rollout + train + ckpt + sync):")
        print(f"  rollout (idle)    : {ar / denom * 100:5.1f}%")
        print(f"  train compute     : {at / denom * 100:5.1f}%")
        print(f"  checkpoint dump   : {ac / denom * 100:5.1f}%")
        print(f"  weight sync       : {asy / denom * 100:5.1f}%")

    print()
    print("note: 'rollout' is the trainer's idle wall-clock waiting for the next")
    print("      streamed chunk; generation overlaps 'train' so it is not a")
    print("      separate serial phase. 'sync' v1 includes one-time engine warmup.")


def write_csv(parsed: ParsedLog, mode: str, out: Path) -> None:
    steps = _completed(parsed, mode)
    subkeys: list[str] = []
    for s in steps:
        for k in s.subphases:
            if k not in subkeys:
                subkeys.append(k)

    def _f(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else ""

    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        if mode == "stream":
            w.writerow(
                ["step", "from_v", "to_v", "total_s", "rollout_s", "train_s",
                 "ckpt_s", "sync_s", "chunks"] + [f"train/{k}" for k in subkeys]
            )
            for s in steps:
                step_no = int(s.metrics.get("step", s.to_v))
                w.writerow(
                    [step_no, s.from_v, s.to_v, _f(s.total_secs), _f(s.inference_secs),
                     _f(s.train_secs), _f(s.ckpt_secs), _f(s.sync_in_step_secs),
                     s.n_chunks]
                    + [f"{s.subphases.get(k, 0.0):.3f}" for k in subkeys]
                )
        else:
            w.writerow(
                ["step", "from_v", "to_v", "total_s", "inference_s", "train_compute_s",
                 "sync_s"] + [f"train/{k}" for k in subkeys]
            )
            for s in steps:
                step_no = int(s.metrics.get("step", s.to_v))
                w.writerow(
                    [step_no, s.from_v, s.to_v, _f(s.total_secs), _f(s.inference_secs),
                     _f(s.train_compute), _f(s.sync_in_step_secs)]
                    + [f"{s.subphases.get(k, 0.0):.3f}" for k in subkeys]
                )
    print(f"\nWrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", nargs="?", default="meshy.log",
                        help="path to the run log (default: meshy.log)")
    parser.add_argument("--log", dest="log_opt", default=None,
                        help="alternative way to pass the log path")
    parser.add_argument("--mode", choices=["auto", "sync", "async", "stream"],
                        default="auto", help="driver mode (default: auto-detect)")
    parser.add_argument("--csv", default=None, help="also write per-step CSV here")
    args = parser.parse_args(argv)

    log_path = Path(args.log_opt or args.log)
    if not log_path.exists():
        print(f"error: log file not found: {log_path}", file=sys.stderr)
        return 1

    parsed = parse_log(log_path)
    mode = args.mode
    if mode == "auto":
        if parsed.has_stream_chunks:
            mode = "stream"
        elif parsed.has_rollout_markers:
            mode = "sync"
        else:
            mode = "async"

    if mode == "sync":
        _compute_sync(parsed.steps, parsed.syncs)
    elif mode == "async":
        _compute_async(parsed.steps, parsed.syncs)
    else:
        _compute_stream(parsed.steps, parsed.syncs)

    print_report(parsed, mode)
    if args.csv:
        write_csv(parsed, mode, Path(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
