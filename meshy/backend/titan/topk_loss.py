"""Student Top-K OPD kernels: Teacher Top-K extraction and the forward KL.

Both sides work on ``[rows, S, V]`` logits in ``S`` chunks so the fp32
transient is bounded by ``[rows, chunk, V]``, the same discipline as
``TitanTrainer._forward_logprobs``.  Under tensor parallelism the LM head
emits a vocab-sharded DTensor; it is gathered here because ``topk`` /
``gather`` along the sharded dim are not TP-aware.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def _dense(logits: torch.Tensor) -> torch.Tensor:
    return logits.full_tensor() if hasattr(logits, "full_tensor") else logits


@torch.no_grad()
def teacher_topk(logits: torch.Tensor, k: int, *, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K ids and their log-probs of ``logits`` ``[rows, S, V]`` -> ``[rows, S, K]``."""
    logits = _dense(logits)
    ids: list[torch.Tensor] = []
    lps: list[torch.Tensor] = []
    for s in range(0, logits.size(1), chunk):
        z = logits[:, s:s + chunk, :].float()
        top = z.topk(k, dim=-1)
        ids.append(top.indices)
        lps.append(top.values - torch.logsumexp(z, dim=-1, keepdim=True))
        del z, top
    return torch.cat(ids, dim=1), torch.cat(lps, dim=1)


def _chunk_gather_logprobs(z: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    z = z.float()
    return z.gather(-1, ids) - torch.logsumexp(z, dim=-1, keepdim=True)


def gather_logprobs(logits: torch.Tensor, ids: torch.Tensor, *, chunk: int = 1024) -> torch.Tensor:
    """``log_softmax(logits).gather(ids)`` without materialising ``[rows, S, V]`` fp32.

    ``ids`` is ``[rows, S, K]``; the result has the same shape in fp32.  Each
    chunk is recomputed in backward (non-reentrant checkpoint) so only the
    bf16 logits stay alive between the passes.
    """
    logits = _dense(logits)
    recompute = torch.is_grad_enabled() and logits.requires_grad
    out: list[torch.Tensor] = []
    for s in range(0, logits.size(1), chunk):
        z = logits[:, s:s + chunk, :]
        i = ids[:, s:s + chunk, :].long()
        if recompute:
            out.append(checkpoint(_chunk_gather_logprobs, z, i, use_reentrant=False))
        else:
            out.append(_chunk_gather_logprobs(z, i))
    return torch.cat(out, dim=1)


def forward_kl_topk(
    student_topk_logprobs: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    *,
    log_prob_min_clamp: float | None = None,
    loss_max_clamp: float | None = None,
) -> torch.Tensor:
    """Per-position ``sum_k p_T(k) (log p_T(k) - log p_S(k))`` over the Teacher's Top-K.

    Same estimator and clamps as VERL's ``compute_forward_kl_topk``: both
    log-prob tensors are clamped from below, the per-token KL from above.
    """
    log_s = student_topk_logprobs.float()
    log_t = teacher_topk_logprobs.float()
    if log_prob_min_clamp is not None:
        log_s = log_s.clamp_min(log_prob_min_clamp)
        log_t = log_t.clamp_min(log_prob_min_clamp)
    kl = (log_t.exp() * (log_t - log_s)).sum(dim=-1)
    if loss_max_clamp is not None:
        kl = kl.clamp(min=-loss_max_clamp, max=loss_max_clamp)
    return kl


__all__ = ["forward_kl_topk", "gather_logprobs", "teacher_topk"]
