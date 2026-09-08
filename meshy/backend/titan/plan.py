"""Dynamic batching plans for the TitanTrainer.

A :class:`Plan` answers two questions for one ``train_step``:

1. Which samples does each data-parallel rank train on (balanced by token
   count instead of by sample count)?
2. How are that rank's samples grouped into mini-batches (one
   ``optimizer.step()`` each) and micro-batches (one forward/backward each),
   and what sequence length does every micro-batch run at?

The planner is a pure function of the sample lengths and the
:class:`PlannerConfig`, so every rank in a job can call :func:`build_plan` on
the same (broadcast) sample list and obtain an identical plan without any
extra communication.

Two forward layouts are supported (see :mod:`meshy.backend.titan.batch`):

``padded``
    Every sample is one row of a ``[rows, S]`` tensor, ``S`` being the
    longest sample in the micro-batch rounded up to ``align``. This keeps the
    plain causal attention kernel (and therefore torchtitan's ring-attention
    context parallelism) usable.

``packed``
    All samples of a micro-batch are concatenated into one ``[1, T]`` row and
    separated by ``cu_seqlens`` for variable-length attention. No padding
    besides the tail rounding of ``T`` to ``align``.

Invariants every consumer relies on
-----------------------------------
* ``len(plan.per_rank[r])`` (number of mini-batches) is the same for every
  rank, and so is ``len(mini.micros)`` for the ``i``-th mini-batch of every
  rank. FSDP issues collectives on every forward/backward, so ranks must run
  the same number of micro-batches; ranks that have fewer real micro-batches
  receive **filler** micro-batches (``sample_idx == ()``) which contribute
  exactly zero to the loss.
* ``MiniPlan.n_docs_global`` / ``n_tokens_global`` are the denominators of
  the loss for that mini-batch, summed over *all* DP ranks, so that FSDP's
  gradient SUM across ranks yields a true global mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

Layout = Literal["padded", "packed"]


@dataclass(frozen=True)
class PlannerConfig:
    layout: Layout
    mini_batch_size: int
    seq_len: int
    align: int
    max_tokens_per_micro: int | None = None
    micro_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.layout not in ("padded", "packed"):
            raise ValueError(f"unknown layout {self.layout!r}")
        if self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be positive")
        if self.align <= 0 or self.seq_len <= 0:
            raise ValueError("align and seq_len must be positive")
        if self.seq_len % self.align != 0:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be a multiple of align ({self.align})"
            )
        if self.max_tokens_per_micro is not None and self.max_tokens_per_micro <= 0:
            raise ValueError("max_tokens_per_micro must be positive")
        if self.micro_batch_size is not None and self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.max_tokens_per_micro is None and self.micro_batch_size is None:
            raise ValueError(
                "either max_tokens_per_micro or micro_batch_size must be set"
            )
        if self.layout == "packed" and self.max_tokens_per_micro is None:
            raise ValueError("packed layout requires max_tokens_per_micro")


@dataclass(frozen=True)
class MicroPlan:
    """One forward/backward pass.

    ``sample_idx`` indexes the rank-local sample list. ``doc_lens[j]`` is the
    number of tokens of ``sample_idx[j]`` that enter the forward (already
    truncated to ``seq_len``). ``seq_len`` is the padded row length
    (``padded``) or the total packed length including tail padding
    (``packed``). An empty ``sample_idx`` marks a filler micro-batch.
    """

    sample_idx: tuple[int, ...]
    seq_len: int
    doc_lens: tuple[int, ...]

    @property
    def is_filler(self) -> bool:
        return not self.sample_idx

    @property
    def n_tokens(self) -> int:
        return sum(self.doc_lens)

    @property
    def n_rows(self) -> int:
        return max(1, len(self.sample_idx))


@dataclass(frozen=True)
class MiniPlan:
    """One optimizer step: its micro-batches and the global loss denominators."""

    micros: tuple[MicroPlan, ...]
    n_docs_global: int
    n_tokens_global: int
    tokens_per_rank: tuple[int, ...]

    @property
    def n_docs_local(self) -> int:
        return sum(len(m.sample_idx) for m in self.micros)


@dataclass(frozen=True)
class Plan:
    """Per-rank mini-batch plans plus the global sample indices each rank owns."""

    per_rank: tuple[tuple[MiniPlan, ...], ...]
    local_indices: tuple[tuple[int, ...], ...]

    @property
    def dp_size(self) -> int:
        return len(self.per_rank)

    def local_samples(self, batch: Sequence, rank: int) -> list:
        """The rank-local sample list that ``MicroPlan.sample_idx`` indexes."""
        return [batch[i] for i in self.local_indices[rank]]


def round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _snake(order: Sequence[int], n_bins: int) -> list[list[int]]:
    """Deal ``order`` into ``n_bins`` lists in boustrophedon order.

    With ``order`` sorted by decreasing length this gives every bin the same
    number of items and near-equal token totals.
    """
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for k, idx in enumerate(order):
        rnd, pos = divmod(k, n_bins)
        bins[pos if rnd % 2 == 0 else n_bins - 1 - pos].append(idx)
    return bins


def _pack_micros(
    local_pos: Sequence[int],
    lengths: Sequence[int],
    cfg: PlannerConfig,
) -> list[MicroPlan]:
    """First-fit-decreasing of ``local_pos`` (rank-local positions) into micro-batches."""
    order = sorted(local_pos, key=lambda p: (-lengths[p], p))
    budget = cfg.max_tokens_per_micro
    # A token budget makes micro_batch_size moot; only enforce the row cap
    # when the user did not ask for a budget at all.
    max_rows = cfg.micro_batch_size if budget is None else None

    # Each open micro: [positions, doc_lens, row_len_or_total]
    micros: list[list] = []
    for p in order:
        L = lengths[p]
        placed = False
        for m in micros:
            if max_rows is not None and len(m[0]) >= max_rows:
                continue
            if budget is not None:
                if cfg.layout == "padded":
                    # Descending order: the first row fixes the row length.
                    cost = (len(m[0]) + 1) * m[2]
                else:
                    cost = round_up(m[2] + L, cfg.align)
                if cost > budget:
                    continue
            m[0].append(p)
            m[1].append(L)
            if cfg.layout == "packed":
                m[2] += L
            placed = True
            break
        if not placed:
            row_len = round_up(L, cfg.align) if cfg.layout == "padded" else L
            micros.append([[p], [L], row_len])

    out: list[MicroPlan] = []
    for positions, doc_lens, total in micros:
        if cfg.layout == "padded":
            seq_len = total
        else:
            seq_len = round_up(total, cfg.align)
        out.append(MicroPlan(tuple(positions), seq_len, tuple(doc_lens)))
    return out


def filler_micro(cfg: PlannerConfig) -> MicroPlan:
    """A zero-weight micro-batch used to equalise micro counts across ranks."""
    return MicroPlan((), cfg.align, ())


def build_plan(
    lengths: Sequence[int],
    loss_tokens: Sequence[int],
    dp_size: int,
    cfg: PlannerConfig,
) -> Plan:
    """Build the plan for one ``train_step`` over the full (global) batch.

    Args:
        lengths: per-sample token count **before** truncation to ``seq_len``.
        loss_tokens: per-sample count of tokens that carry loss after the
            shift-by-one and truncation (``sum(mask_assistant[1:L])``); used
            for the token-mean denominator and for DP balancing diagnostics.
        dp_size: number of data-parallel ranks (``batch`` mesh size).
        cfg: planner configuration.
    """
    n = len(lengths)
    if n == 0:
        raise ValueError("cannot plan an empty batch")
    if len(loss_tokens) != n:
        raise ValueError("lengths and loss_tokens must have the same size")
    if dp_size <= 0 or n % dp_size != 0:
        raise ValueError(
            f"batch size {n} is not divisible by DP size {dp_size}; adjust "
            "rollout.batch_size * group_size to a multiple of dp_replicate * dp_shard"
        )
    trunc = [min(int(L), cfg.seq_len) for L in lengths]

    # 1. DP partition: equal sample counts, balanced tokens.
    order = sorted(range(n), key=lambda i: (-trunc[i], i))
    rank_samples = _snake(order, dp_size)
    n_local = n // dp_size
    n_mini = (n_local + cfg.mini_batch_size - 1) // cfg.mini_batch_size

    # 2. Per rank: deal samples into mini-batches (snake again so every
    #    optimizer step sees a mix of lengths), then pack micro-batches.
    per_rank_minis: list[list[list[MicroPlan]]] = []
    per_rank_mini_members: list[list[list[int]]] = []
    for r in range(dp_size):
        local = rank_samples[r]  # global indices, descending length
        local_len = [trunc[g] for g in local]
        local_order = sorted(range(n_local), key=lambda p: (-local_len[p], p))
        mini_members = _snake(local_order, n_mini)
        per_rank_mini_members.append(mini_members)
        per_rank_minis.append(
            [_pack_micros(members, local_len, cfg) for members in mini_members]
        )

    # 3. Equalise micro counts per mini-batch index and attach global denominators.
    per_rank: list[tuple[MiniPlan, ...]] = []
    for r in range(dp_size):
        minis: list[MiniPlan] = []
        for i in range(n_mini):
            n_micro = max(len(per_rank_minis[q][i]) for q in range(dp_size))
            micros = list(per_rank_minis[r][i])
            micros.extend(filler_micro(cfg) for _ in range(n_micro - len(micros)))
            n_docs_global = sum(len(per_rank_mini_members[q][i]) for q in range(dp_size))
            n_tokens_global = 0
            tokens_per_rank = []
            for q in range(dp_size):
                members = per_rank_mini_members[q][i]
                globals_q = [rank_samples[q][p] for p in members]
                n_tokens_global += sum(int(loss_tokens[g]) for g in globals_q)
                tokens_per_rank.append(sum(trunc[g] for g in globals_q))
            minis.append(
                MiniPlan(
                    micros=tuple(micros),
                    n_docs_global=n_docs_global,
                    n_tokens_global=n_tokens_global,
                    tokens_per_rank=tuple(tokens_per_rank),
                )
            )
        per_rank.append(tuple(minis))

    return Plan(
        per_rank=tuple(per_rank),
        local_indices=tuple(tuple(s) for s in rank_samples),
    )


def plan_stats(minis: Sequence[MiniPlan], layout: Layout) -> dict[str, float]:
    """Diagnostics for one rank's plan (padding ratio, imbalance, ...)."""
    n_micro = 0
    real_tokens = 0
    forward_tokens = 0
    for mini in minis:
        for m in mini.micros:
            n_micro += 1
            real_tokens += m.n_tokens
            rows = m.n_rows if layout == "padded" else 1
            forward_tokens += rows * m.seq_len
    tokens_per_rank = [t for mini in minis for t in mini.tokens_per_rank]
    mean_tpr = sum(tokens_per_rank) / max(1, len(tokens_per_rank))
    imbalance = (max(tokens_per_rank) / mean_tpr) if mean_tpr > 0 else 1.0
    return {
        "n_micro": float(n_micro),
        "tokens_per_micro": real_tokens / max(1, n_micro),
        "padding_ratio": 1.0 - real_tokens / max(1, forward_tokens),
        "dp_token_imbalance": float(imbalance),
    }


def resolve_align(seq_bucket: int, divisor: int, seq_len: int) -> int:
    """Sequence-length rounding granularity satisfying both the bucket and
    the parallelism divisor (``tp * cp * 2``), capped at ``seq_len``."""
    align = math.lcm(max(1, seq_bucket), max(1, divisor))
    if align >= seq_len:
        align = seq_len
    if seq_len % align != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be a multiple of the sequence alignment "
            f"({align} = lcm(seq_bucket={seq_bucket}, divisor={divisor}))"
        )
    return align


__all__ = [
    "Layout",
    "MicroPlan",
    "MiniPlan",
    "Plan",
    "PlannerConfig",
    "build_plan",
    "filler_micro",
    "plan_stats",
    "resolve_align",
    "round_up",
]
