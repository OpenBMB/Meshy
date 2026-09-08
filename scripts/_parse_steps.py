"""Parse per-step timing from a quick_test / tq_quick_test run log.

Reads a loguru log (default INFO format, leading ``YYYY-MM-DD HH:MM:SS.mmm``
timestamp) and reports the wall time of steps ``--lo``..``--hi`` (inclusive)
two independent ways:

* ``step_total`` summed from the pipeline's own ``step N (timing)`` lines
  (CUDA-synced, the authoritative number); and
* timestamp delta between the step-``lo`` and step-``hi`` ``(timing)`` lines
  (a cross-check that includes any gaps the in-code timer doesn't cover).

Usage::

    python scripts/_parse_steps.py --log run.log --lo 1 --hi 10 --label normal
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime

# loguru default sink: "2026-06-12 06:50:01.123 | INFO     | ... - message"
TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
# "step 7 (timing) | step_total=12.345s  rollout=9.1s  train=2.0s ..."
TIMING = re.compile(r"step (\d+) \(timing\) \| (.*)$")
STEP_TOTAL = re.compile(r"step_total=([0-9.]+)s")


def _ts(line: str) -> datetime | None:
    m = TS.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--lo", type=int, default=1, help="first step (inclusive)")
    ap.add_argument("--hi", type=int, default=10, help="last step (inclusive)")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    # step -> (timestamp, step_total seconds)
    steps: dict[int, tuple[datetime | None, float | None]] = {}
    with open(args.log, errors="replace") as fh:
        for line in fh:
            m = TIMING.search(line)
            if not m:
                continue
            step = int(m.group(1))
            tot = STEP_TOTAL.search(m.group(2))
            steps[step] = (_ts(line), float(tot.group(1)) if tot else None)

    present = sorted(steps)
    print(f"\n===== {args.label} =====")
    print(f"steps with (timing) lines: {present[:3]}..{present[-3:] if len(present) > 3 else present}"
          f"  (count={len(present)})")

    wanted = [s for s in range(args.lo, args.hi + 1) if s in steps]
    missing = [s for s in range(args.lo, args.hi + 1) if s not in steps]
    if missing:
        print(f"!! MISSING steps in [{args.lo},{args.hi}]: {missing}")
    if not wanted:
        print("!! no step_total data in requested range")
        return 1

    # Method 1: sum of in-code step_total.
    totals = [steps[s][1] for s in wanted if steps[s][1] is not None]
    sum_total = sum(totals) if totals else float("nan")

    # Method 2: timestamp delta lo..hi (wall clock across the same steps).
    ts_lo = steps[wanted[0]][0]
    ts_hi = steps[wanted[-1]][0]
    # The (timing) line is emitted at the END of a step, so the delta between
    # step lo and step hi covers steps (lo+1)..hi. Add step lo's own step_total
    # to recover the full lo..hi span.
    wall_delta = (ts_hi - ts_lo).total_seconds() if (ts_lo and ts_hi) else float("nan")
    first_total = steps[wanted[0]][1] or 0.0
    wall_span = wall_delta + first_total

    print(f"per-step step_total (s): " +
          "  ".join(f"{s}={steps[s][1]:.2f}" if steps[s][1] is not None else f"{s}=?"
                    for s in wanted))
    print(f"\nMethod 1 (sum of in-code step_total, steps {wanted[0]}..{wanted[-1]}, "
          f"n={len(totals)}): {sum_total:.2f}s")
    print(f"Method 2 (wall-clock {wanted[0]}->{wanted[-1]} incl. step {wanted[0]}): "
          f"{wall_span:.2f}s  (raw delta {wall_delta:.2f}s)")
    if totals:
        print(f"mean/step: {sum_total/len(totals):.2f}s   "
              f"min={min(totals):.2f}s  max={max(totals):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
