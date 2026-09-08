"""Analyze per-step average reward and timing from Meshy trajectory logs.

Reads ``.xrl_runtime/*/trajectories.jsonl`` (or ``.xrl-runtime``) line by line.
Each jsonl record carries a ``timestamp``, a ``round`` field (training step) and
a ``reward``. Because ``trajectory`` can be very large, we extract only those
few top-level fields via regex instead of parsing the full JSON object.

Besides the per-step average reward, we also report the wall-clock gap between
consecutive steps, measured as the difference between the earliest sample
timestamp of each step.

Usage::

    python scripts/analyze_trajectory_rewards.py
    python scripts/analyze_trajectory_rewards.py --root .xrl_runtime/repro_async2
    python scripts/analyze_trajectory_rewards.py --csv rewards.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ``timestamp`` and ``round`` sit at the very start of the line (before the huge
# ``trajectory`` blob), so the first regex match is the real top-level value.
_TIMESTAMP_RE = re.compile(r'"timestamp"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
_ROUND_RE = re.compile(r'"round"\s*:\s*(-?\d+)')
# ``reward`` is a top-level field emitted *after* the trajectory, so a stray
# ``"reward": ...`` inside the chat content would match first. Take the LAST
# match to reliably pick the top-level reward (only ``advantage`` follows it).
_REWARD_RE = re.compile(r'"reward"\s*:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)')


@dataclass
class StepStats:
    count: int = 0
    reward_sum: float = 0.0
    first_ts: float | None = None  # earliest sample timestamp seen for this step
    last_ts: float | None = None  # latest sample timestamp seen for this step

    def add(self, reward: float, ts: float | None = None) -> None:
        self.count += 1
        self.reward_sum += reward
        if ts is not None:
            if self.first_ts is None or ts < self.first_ts:
                self.first_ts = ts
            if self.last_ts is None or ts > self.last_ts:
                self.last_ts = ts

    @property
    def avg_reward(self) -> float:
        return self.reward_sum / self.count if self.count else float("nan")


@dataclass
class RunStats:
    path: Path
    steps: dict[int, StepStats] = field(default_factory=dict)
    bad_lines: int = 0

    def add(self, step: int, reward: float, ts: float | None = None) -> None:
        self.steps.setdefault(step, StepStats()).add(reward, ts)

    @property
    def total_samples(self) -> int:
        return sum(s.count for s in self.steps.values())

    @property
    def overall_avg(self) -> float:
        total = self.total_samples
        if not total:
            return float("nan")
        return sum(s.reward_sum for s in self.steps.values()) / total


def _discover_logs(root: Path) -> list[Path]:
    """Return ``trajectories.jsonl`` files under *root*."""
    if root.is_file() and root.name == "trajectories.jsonl":
        return [root]
    if root.is_dir():
        direct = root / "trajectories.jsonl"
        if direct.is_file():
            return [direct]
        return sorted(root.glob("*/trajectories.jsonl"))
    return []


def _last_float(pattern: re.Pattern[str], line: str) -> float | None:
    """Return the last regex match's group as float, or None."""
    last = None
    for last in pattern.finditer(line):
        pass
    return float(last.group(1)) if last else None


def _parse_line(line: str) -> tuple[int, float, float | None] | None:
    round_m = _ROUND_RE.search(line)  # first match: top-level round (line head)
    if not round_m:
        return None
    reward = _last_float(_REWARD_RE, line)  # last match: top-level reward
    if reward is None:
        return None
    ts_m = _TIMESTAMP_RE.search(line)  # first match: top-level timestamp
    ts = float(ts_m.group(1)) if ts_m else None
    return int(round_m.group(1)), reward, ts


def analyze_file(path: Path) -> RunStats:
    stats = RunStats(path=path)
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parsed = _parse_line(line)
            if parsed is None:
                stats.bad_lines += 1
                print(f"warning: {path}:{lineno}: cannot parse round/reward", file=sys.stderr)
                continue
            step, reward, ts = parsed
            stats.add(step, reward, ts)
    return stats


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]


def _print_run(stats: RunStats) -> None:
    print(f"\n===== {stats.path} =====")
    if stats.bad_lines:
        print(f"skipped malformed lines: {stats.bad_lines}")
    if not stats.steps:
        print("no valid records")
        return

    print(
        f"samples={stats.total_samples}  steps={len(stats.steps)}  "
        f"overall_avg_reward={stats.overall_avg:.6f}"
    )
    print(f"{'step':>6}  {'count':>8}  {'avg_reward':>12}  {'sum_reward':>12}")
    for step in sorted(stats.steps):
        s = stats.steps[step]
        print(f"{step:>6}  {s.count:>8}  {s.avg_reward:>12.6f}  {s.reward_sum:>12.4f}")

    _print_timing(stats)


def _step_gap(stats: RunStats, step: int, prev_real_ts: float | None) -> float | None:
    """Gap (s) between this step's earliest ts and the previous *real* step's.

    Negative rounds (e.g. ``-1``) are async-abort sentinels rather than real
    training steps, so they never anchor a gap.
    """
    first_ts = stats.steps[step].first_ts
    if step < 0 or prev_real_ts is None or first_ts is None:
        return None
    return first_ts - prev_real_ts


def _print_timing(stats: RunStats) -> None:
    """Wall-clock gap between consecutive steps' earliest sample timestamps."""
    print("\n-- per-step timing (gap = earliest-sample ts diff vs previous step) --")
    print("   negative rounds are async-abort sentinels, excluded from gaps")
    print(f"{'step':>6}  {'earliest_ts':>16}  {'gap_from_prev(s)':>18}")
    prev_real_ts: float | None = None
    deltas: list[float] = []
    for step in sorted(stats.steps):
        first_ts = stats.steps[step].first_ts
        gap = _step_gap(stats, step, prev_real_ts)
        if gap is not None:
            deltas.append(gap)
            gap_str = f"{gap:.3f}"
        else:
            gap_str = "-"
        print(f"{step:>6}  {_fmt_ts(first_ts):>16}  {gap_str:>18}")
        if step >= 0 and first_ts is not None:
            prev_real_ts = first_ts
    if deltas:
        print(
            f"gap stats (s): mean={sum(deltas) / len(deltas):.3f}  "
            f"min={min(deltas):.3f}  max={max(deltas):.3f}  total={sum(deltas):.3f}"
        )


def _write_csv(rows: list[tuple[str, int, int, float, float, str, str]], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["run", "step", "count", "avg_reward", "sum_reward", "earliest_ts", "gap_from_prev_s"]
        )
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-step average reward from trajectories.jsonl")
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="runtime root, a run dir, or a trajectories.jsonl file "
        "(default: search .xrl_runtime and .xrl-runtime under cwd)",
    )
    ap.add_argument("--csv", type=Path, default=None, help="optional CSV output path")
    args = ap.parse_args()

    if args.root is not None:
        roots = [args.root.expanduser().resolve()]
    else:
        cwd = Path.cwd()
        roots = [p for name in (".xrl_runtime", ".xrl-runtime") if (p := cwd / name).is_dir()]

    logs: list[Path] = []
    for root in roots:
        logs.extend(_discover_logs(root))

    if not logs:
        print("error: no trajectories.jsonl found", file=sys.stderr)
        return 1

    csv_rows: list[tuple[str, int, int, float, float, str, str]] = []
    all_steps: dict[int, StepStats] = defaultdict(StepStats)

    for path in logs:
        stats = analyze_file(path)
        _print_run(stats)
        run_name = str(path.parent.name if path.name == "trajectories.jsonl" else path)
        prev_real_ts: float | None = None
        for step in sorted(stats.steps):
            s = stats.steps[step]
            gap = _step_gap(stats, step, prev_real_ts)
            gap_str = f"{gap:.3f}" if gap is not None else ""
            csv_rows.append(
                (run_name, step, s.count, s.avg_reward, s.reward_sum, _fmt_ts(s.first_ts), gap_str)
            )
            if step >= 0 and s.first_ts is not None:
                prev_real_ts = s.first_ts
            all_steps[step].count += s.count
            all_steps[step].reward_sum += s.reward_sum

    if len(logs) > 1 and all_steps:
        total = sum(s.count for s in all_steps.values())
        overall = sum(s.reward_sum for s in all_steps.values()) / total
        print(f"\n===== ALL RUNS ({len(logs)} files) =====")
        print(f"samples={total}  steps={len(all_steps)}  overall_avg_reward={overall:.6f}")
        print(f"{'step':>6}  {'count':>8}  {'avg_reward':>12}")
        for step in sorted(all_steps):
            s = all_steps[step]
            print(f"{step:>6}  {s.count:>8}  {s.avg_reward:>12.6f}")

    if args.csv is not None:
        _write_csv(csv_rows, args.csv.expanduser().resolve())
        print(f"\nCSV written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
