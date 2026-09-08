"""Micro-batch construction for the TitanTrainer.

:func:`build_micro_batch` turns one :class:`~meshy.backend.titan.plan.MicroPlan`
plus the rank-local sample list into a :class:`Batch` ready for the model
forward. It does three jobs so no downstream code has to repeat them:

1. **Lay out** the samples in the requested layout:

   * ``padded``: one sample per row of ``[rows, S]`` tensors, tail-padded to
     ``S = micro.seq_len``. Positions are ``arange(S)``; no attention mask is
     needed because padding sits after the causal frontier of every real token.
   * ``packed``: all samples concatenated into a single ``[1, T]`` row,
     ``T = micro.seq_len``, plus a :class:`VarlenMetadata` with the document
     boundaries and per-document RoPE positions restarting at 0. The tail
     padding (if any) is its own document so it never leaks into a sample.

2. **Shift-by-one** the next-token label, the assistant mask and the rollout
   log-probs *inside each sample* so they align with the forward logprob
   convention (``new_lp[t] = log p(input_ids[t+1] | <=t)``). The last position
   of every sample gets label 0 / mask 0.

3. **CP-shard** every per-token tensor through :class:`CpSharder` in a single
   call (``padded`` only; ``packed`` + CP is rejected upstream) so they share
   one head-tail permutation and stay token-for-token aligned.

Every per-token tensor carries a ``doc_ids`` companion mapping each position
to its sample (``0 .. n_docs-1``) or to the padding slot ``n_docs``. The loss
uses it for per-sequence reductions in both layouts, so the trainer never
branches on the layout.

Convention for ``rollout_logprobs``
-----------------------------------
SGLang reports ``logprobs[t] = log p(token_t | tokens[<t])`` for the
generated assistant tokens. After the same shift-by-one as the mask, element
``[t]`` becomes the next-token log-prob aligned with ``new_lp[t]`` — directly
usable as the PPO behavior policy ``old_lp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch

from .plan import Layout, MicroPlan

if TYPE_CHECKING:
    from .cp import CpSharder


@dataclass
class Batch:
    """One micro-batch in CP-local layout, next-token aligned."""

    input_ids: torch.Tensor              # [rows, S_local] long
    labels: torch.Tensor                 # [rows, S_local] long  (next-token, shifted, pad 0)
    positions: torch.Tensor              # [rows, S_local] int32 (RoPE positions)
    mask: torch.Tensor                   # [rows, S_local] float (assistant next-token, shifted)
    doc_ids: torch.Tensor                # [rows, S_local] long  (sample index; n_docs = padding)
    advantages: torch.Tensor             # [n_docs] float
    rollout_logprobs: torch.Tensor | None  # [rows, S_local] float | None
    lengths: torch.Tensor                # [n_docs] int32, tokens per sample entering the forward
    n_docs: int                          # number of real samples
    attention_masks: Any = None          # VarlenMetadata for ``packed``; None for ``padded``

    @property
    def n_slots(self) -> int:
        """Segment-buffer size: one slot per sample plus one for padding."""
        return self.n_docs + 1


def _sample_view(td: Any, L: int, need_rollout_lp: bool):
    tokens = td["tokens"][:L].to(dtype=torch.long)
    mask = td["mask_assistant"][:L].to(dtype=torch.float32)
    lp = td["logprobs"][:L].to(dtype=torch.float32) if need_rollout_lp else None
    return tokens, mask, lp


def _shift(x: torch.Tensor) -> torch.Tensor:
    """``x[1:] + [0]``: align a per-token tensor with the next-token logprob."""
    return torch.cat([x[1:], x.new_zeros(1)])


def build_micro_batch(
    samples: Sequence[Any],
    micro: MicroPlan,
    *,
    layout: Layout,
    device: torch.device,
    sharder: "CpSharder",
    need_rollout_lp: bool,
) -> Batch:
    """Materialise ``micro`` from the rank-local ``samples`` as a :class:`Batch`."""
    n_docs = len(micro.sample_idx)
    advantages = torch.zeros(n_docs, dtype=torch.float32)
    lengths = torch.tensor(micro.doc_lens, dtype=torch.int32)
    for j, idx in enumerate(micro.sample_idx):
        advantages[j] = float(samples[idx].get("advantage", 0.0))

    if layout == "padded":
        rows, S = micro.n_rows, micro.seq_len
        input_ids = torch.zeros(rows, S, dtype=torch.long)
        labels = torch.zeros(rows, S, dtype=torch.long)
        mask = torch.zeros(rows, S, dtype=torch.float32)
        doc_ids = torch.arange(rows, dtype=torch.long).unsqueeze(1).expand(rows, S).clone()
        rollout_lp = torch.zeros(rows, S, dtype=torch.float32) if need_rollout_lp else None
        for j, idx in enumerate(micro.sample_idx):
            L = micro.doc_lens[j]
            tokens, m, lp = _sample_view(samples[idx], L, need_rollout_lp)
            input_ids[j, :L] = tokens
            labels[j, :L] = _shift(tokens)
            mask[j, :L] = _shift(m)
            if rollout_lp is not None:
                rollout_lp[j, :L] = _shift(lp)
        if n_docs == 0:
            # Filler micro: the single row is padding.
            doc_ids.fill_(0)
        positions = torch.arange(S, dtype=torch.int32).unsqueeze(0).expand(rows, S).contiguous()
        attention_masks = None
    elif layout == "packed":
        T = micro.seq_len
        used = micro.n_tokens
        pad = T - used
        assert pad >= 0, f"micro seq_len {T} smaller than its tokens {used}"
        input_ids = torch.zeros(1, T, dtype=torch.long)
        labels = torch.zeros(1, T, dtype=torch.long)
        mask = torch.zeros(1, T, dtype=torch.float32)
        doc_ids = torch.full((1, T), n_docs, dtype=torch.long)
        positions = torch.zeros(1, T, dtype=torch.int32)
        rollout_lp = torch.zeros(1, T, dtype=torch.float32) if need_rollout_lp else None
        cu = [0]
        off = 0
        for j, idx in enumerate(micro.sample_idx):
            L = micro.doc_lens[j]
            tokens, m, lp = _sample_view(samples[idx], L, need_rollout_lp)
            input_ids[0, off:off + L] = tokens
            labels[0, off:off + L] = _shift(tokens)
            mask[0, off:off + L] = _shift(m)
            doc_ids[0, off:off + L] = j
            positions[0, off:off + L] = torch.arange(L, dtype=torch.int32)
            if rollout_lp is not None:
                rollout_lp[0, off:off + L] = _shift(lp)
            off += L
            cu.append(off)
        doc_lens = list(micro.doc_lens)
        if pad > 0:
            positions[0, off:] = torch.arange(pad, dtype=torch.int32)
            cu.append(T)
            doc_lens.append(pad)
        attention_masks = make_varlen_metadata(cu, max(doc_lens), device)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    positions = positions.to(device)
    mask = mask.to(device)
    doc_ids = doc_ids.to(device)
    advantages = advantages.to(device)
    lengths = lengths.to(device)
    if rollout_lp is not None:
        rollout_lp = rollout_lp.to(device)

    if layout == "padded":
        # One CP shard for all per-token tensors so they receive identical
        # head-tail permutations. Optional tensors go last, unpacked by count.
        optional = [rollout_lp] if rollout_lp is not None else []
        sharded = sharder.shard_seq(
            input_ids, labels, positions, mask, doc_ids, *optional
        )
        input_ids, labels, positions, mask, doc_ids = sharded[:5]
        if rollout_lp is not None:
            rollout_lp = sharded[5]

    return Batch(
        input_ids=input_ids,
        labels=labels,
        positions=positions,
        mask=mask,
        doc_ids=doc_ids,
        advantages=advantages,
        rollout_logprobs=rollout_lp,
        lengths=lengths,
        n_docs=n_docs,
        attention_masks=attention_masks,
    )


def make_varlen_metadata(cu_seqlens: Sequence[int], max_len: int, device: torch.device):
    """Build torchtitan's ``VarlenMetadata`` from explicit document boundaries.

    torchtitan's own ``create_varlen_metadata_for_document`` derives the
    boundaries from EOS tokens, which is wrong for RL samples (multi-turn
    chats contain EOS after every turn; truncated samples end without one).
    """
    from torchtitan.models.common.attention import VarlenMetadata

    cu = torch.tensor(list(cu_seqlens), dtype=torch.int32, device=device)
    return VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=int(max_len), max_k=int(max_len))


__all__ = ["Batch", "build_micro_batch", "make_varlen_metadata"]
