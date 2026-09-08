"""Student Top-K OPD kernels against dense references (CPU)."""

import torch

from meshy.backend.titan.topk_loss import forward_kl_topk, gather_logprobs, teacher_topk


def _logits(rows=2, S=7, V=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, S, V, generator=g) * 3


def test_teacher_topk_matches_dense_log_softmax():
    z = _logits()
    ids, lps = teacher_topk(z, 5, chunk=3)
    ref_lp = torch.log_softmax(z, dim=-1)
    ref = ref_lp.topk(5, dim=-1)
    assert torch.equal(ids, ref.indices)
    assert torch.allclose(lps, ref.values, atol=1e-6)
    assert ids.shape == (2, 7, 5) and lps.dtype == torch.float32


def test_gather_logprobs_matches_dense_and_propagates_grad():
    z = _logits().requires_grad_(True)
    ids = torch.randint(0, 50, (2, 7, 4))
    out = gather_logprobs(z, ids, chunk=2)
    ref = torch.log_softmax(z, dim=-1).gather(-1, ids)
    assert torch.allclose(out, ref, atol=1e-6)
    out.sum().backward()
    g1 = z.grad.clone()
    z.grad = None
    ref.sum().backward()
    assert torch.allclose(g1, z.grad, atol=1e-6)


def test_forward_kl_topk_formula_and_clamps():
    log_t = torch.log(torch.tensor([[[0.5, 0.3, 0.2]]]))
    log_s = torch.log(torch.tensor([[[0.2, 0.3, 0.5]]]))
    kl = forward_kl_topk(log_s, log_t)
    ref = (log_t.exp() * (log_t - log_s)).sum(-1)
    assert torch.allclose(kl, ref)
    # identical distributions -> zero
    assert torch.allclose(forward_kl_topk(log_t, log_t), torch.zeros(1, 1))
    # a vanishing student prob is bounded by the log-prob clamp, then the loss clamp
    tiny = torch.log(torch.tensor([[[1e-30, 0.3, 0.5]]]))
    unclamped = forward_kl_topk(tiny, log_t)
    clamped = forward_kl_topk(tiny, log_t, log_prob_min_clamp=-10.0)
    assert unclamped.item() > clamped.item() > 0
    assert forward_kl_topk(tiny, log_t, loss_max_clamp=1.0).item() == 1.0
