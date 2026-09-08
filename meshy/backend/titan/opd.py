"""Student Top-K OPD on the Titan trainer: Teacher-column layout and the loss.

:func:`layout_teacher_topk` mirrors :func:`~meshy.backend.titan.batch.build_micro_batch`
for one extra per-token tensor of shape ``[L, K]``. Teacher rows are indexed
by logits position, so no shift-by-one is applied: ``teacher[t]`` already
describes the distribution of ``labels[t]``.

:class:`StudentTopKTrainer` overrides three hooks of
:class:`~meshy.backend.titan.TitanTrainer`:

* ``_build_micro`` attaches the Teacher columns to the micro-batch;
* ``_run_mini_batch`` replaces the PPO forward/loss with the distillation
  forward/loss (no behaviour-policy snapshot is needed);
* ``_reduce_mini_metrics`` reports distillation metrics instead of PPO ones.

Planning, layouts, CP sharding, gradient clipping, the optimizer step, the LR
schedule, checkpointing, rollout metrics and TensorBoard publication are all
inherited unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torchtitan.distributed import utils as dist_utils

from meshy.config import OPD_TEACHER_FIELDS

from .batch import Batch
from .metrics import TitanTrainer
from .plan import Layout, MicroPlan, MiniPlan
from .topk_loss import forward_kl_topk, gather_logprobs


# ── Teacher columns in the micro-batch layout ───────────────────────────
def infer_top_k(samples: Sequence[Any], default: int = 1) -> int:
    for td in samples:
        value = td.get(OPD_TEACHER_FIELDS[0], None)
        if value is not None:
            return int(value.shape[-1])
    return default


def _teacher_view(td: Any, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = td[OPD_TEACHER_FIELDS[0]]
    lps = td[OPD_TEACHER_FIELDS[1]]
    if ids.shape[0] < L or lps.shape[0] < L:
        raise ValueError(
            f"teacher columns cover {int(ids.shape[0])} positions but the student "
            f"micro-batch needs {L}; the Teacher seq_len must be >= the Student's"
        )
    return ids[:L].to(dtype=torch.long), lps[:L].to(dtype=torch.float32)


def layout_teacher_topk(
    samples: Sequence[Any],
    micro: MicroPlan,
    *,
    layout: Layout,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, S, K]`` Teacher ids / log-probs in the micro-batch's layout."""
    k = infer_top_k([samples[i] for i in micro.sample_idx])
    if layout == "padded":
        rows, S = micro.n_rows, micro.seq_len
        ids = torch.zeros(rows, S, k, dtype=torch.long)
        lps = torch.zeros(rows, S, k, dtype=torch.float32)
        for j, idx in enumerate(micro.sample_idx):
            L = micro.doc_lens[j]
            i, l = _teacher_view(samples[idx], L)
            ids[j, :L] = i
            lps[j, :L] = l
    elif layout == "packed":
        T = micro.seq_len
        ids = torch.zeros(1, T, k, dtype=torch.long)
        lps = torch.zeros(1, T, k, dtype=torch.float32)
        off = 0
        for j, idx in enumerate(micro.sample_idx):
            L = micro.doc_lens[j]
            i, l = _teacher_view(samples[idx], L)
            ids[0, off:off + L] = i
            lps[0, off:off + L] = l
            off += L
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return ids.to(device), lps.to(device)


def split_teacher_topk(
    values: torch.Tensor, micro: MicroPlan, *, layout: Layout
) -> list[torch.Tensor]:
    """Inverse of :func:`layout_teacher_topk` for one ``[rows, S, K]`` tensor.

    Returns one ``[L_j, K]`` tensor per ``micro.sample_idx[j]``.
    """
    out: list[torch.Tensor] = []
    if layout == "padded":
        for j, L in enumerate(micro.doc_lens):
            out.append(values[j, :L])
    elif layout == "packed":
        off = 0
        for L in micro.doc_lens:
            out.append(values[0, off:off + L])
            off += L
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return out



# ── Student trainer ─────────────────────────────────────────────────────
@dataclass
class TopKBatch(Batch):
    """A :class:`Batch` plus the Teacher's Top-K columns in the same layout."""

    teacher_topk_ids: torch.Tensor | None = None       # [rows, S_local, K] long
    teacher_topk_logprobs: torch.Tensor | None = None  # [rows, S_local, K] float


class StudentTopKTrainer(TitanTrainer):
    def __init__(
        self,
        *args: Any,
        log_prob_min_clamp: float | None = -10.0,
        loss_max_clamp: float | None = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.log_prob_min_clamp = log_prob_min_clamp
        self.loss_max_clamp = loss_max_clamp

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------
    def _build_micro(self, samples: Sequence[Any], micro, *, need_rollout_lp: bool | None = None) -> TopKBatch:
        mb = super()._build_micro(samples, micro, need_rollout_lp=need_rollout_lp)
        ids, lps = layout_teacher_topk(samples, micro, layout=self.batch_layout, device=self.device)
        if self.batch_layout == "padded":
            # Same CP head-tail permutation as the base tensors: the sharder is
            # deterministic in (sequence length, load balancer).
            ids, lps = self.sharder.shard_seq(ids, lps)
        return TopKBatch(**vars(mb), teacher_topk_ids=ids, teacher_topk_logprobs=lps)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _distill_loss(self, mb: TopKBatch, mini: MiniPlan) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.model_parts[0](mb.input_ids, positions=mb.positions, attention_masks=mb.attention_masks)
        k = mb.teacher_topk_ids.shape[-1]
        # One gather for the Teacher's K candidates plus the sampled token, so
        # the train-vs-rollout diagnostic costs no extra logsumexp.
        ids = torch.cat([mb.teacher_topk_ids, mb.labels.unsqueeze(-1)], dim=-1)
        gathered = gather_logprobs(logits, ids, chunk=self._LOGPROB_CHUNK)
        del logits
        student_topk = gathered[..., :k]
        sampled_lp = gathered[..., k].detach()

        per_token = forward_kl_topk(
            student_topk, mb.teacher_topk_logprobs,
            log_prob_min_clamp=self.log_prob_min_clamp, loss_max_clamp=self.loss_max_clamp,
        )
        mask = mb.mask
        per_token = per_token * mask

        # Same aggregation as the PPO loss: sequence mean by default, token
        # mean with ``calculate_per_token_loss``; denominators are global.
        flat_ids = mb.doc_ids.reshape(-1)
        seg_sum = per_token.new_zeros(mb.n_slots).index_add_(0, flat_ids, per_token.reshape(-1))
        seg_cnt = mask.new_zeros(mb.n_slots).index_add_(0, flat_ids, mask.reshape(-1))
        seg_cnt_global = self.sharder.all_reduce_sum(seg_cnt)
        n = mb.n_docs
        if self.calculate_per_token_loss:
            loss = seg_sum[:n].sum() / max(1, mini.n_tokens_global)
        else:
            loss = (seg_sum[:n] / seg_cnt_global[:n].clamp(min=1)).sum() / max(1, mini.n_docs_global)

        with torch.no_grad():
            sums = {
                "loss": loss.detach(),
                "token_count": mask.sum(),
                "student_mass_sum": (student_topk.detach().exp().sum(-1) * mask).sum(),
                "teacher_mass_sum": (mb.teacher_topk_logprobs.exp().sum(-1) * mask).sum(),
            }
            if mb.rollout_logprobs is not None:
                lr = (mb.rollout_logprobs - sampled_lp).clamp(-10.0, 10.0)
                k3 = lr.exp() - lr - 1.0
                sums["train_rollout_kl_sum"] = (torch.nan_to_num(k3) * mask).sum()
                sums["train_rollout_logdiff_abs_sum"] = (
                    torch.nan_to_num((mb.rollout_logprobs - sampled_lp).abs()) * mask
                ).sum()
        return loss, sums

    # ------------------------------------------------------------------
    # Mini-batch loop
    # ------------------------------------------------------------------
    def _run_mini_batch(self, samples: Sequence[Any], mini: MiniPlan, timer) -> tuple[dict[str, float], torch.Tensor]:
        sums: dict[str, torch.Tensor] | None = None
        self.optimizers.zero_grad()
        for micro in mini.micros:
            with timer.timer("train/pack_batch", sync=True):
                mb = self._build_micro(samples, micro)
            with self.train_context():
                with timer.timer("train/forward", sync=True):
                    loss, mb_sums = self._distill_loss(mb, mini)
                with timer.timer("train/backward", sync=True):
                    loss.backward()
            sums = mb_sums if sums is None else {k: sums[k] + v for k, v in mb_sums.items()}
            del mb, loss, mb_sums

        with timer.timer("train/empty_cache", sync=True):
            torch.cuda.empty_cache()
        with timer.timer("train/clip_grad_norm", sync=True):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for part in self.model_parts for p in part.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                ep_enabled=self.parallel_dims.ep_enabled,
            )
        with timer.timer("train/optim_step", sync=True):
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
        assert sums is not None
        return self._reduce_mini_metrics(sums), grad_norm

    def _reduce_mini_metrics(self, sums: dict[str, torch.Tensor]) -> dict[str, float]:
        keys = list(sums)
        stacked = torch.stack([sums[k].float() for k in keys])
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")
        if loss_mesh is not None:
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=loss_mesh.get_group())
        values = dict(zip(keys, stacked.tolist()))
        tokens = max(values["token_count"], 1.0)
        out = {
            "train/distill_loss": values["loss"],
            "train/student_topk_mass": values["student_mass_sum"] / tokens,
            "train/teacher_topk_mass": values["teacher_mass_sum"] / tokens,
        }
        if "train_rollout_kl_sum" in values:
            # Picked up by the inherited metrics mapping under ``train/``.
            out["train_rollout_kl"] = values["train_rollout_kl_sum"] / tokens
            out["train_rollout_logdiff_abs"] = values["train_rollout_logdiff_abs_sum"] / tokens
        return out


__all__ = [
    "StudentTopKTrainer",
    "TopKBatch",
    "infer_top_k",
    "layout_teacher_topk",
    "split_teacher_topk",
]
