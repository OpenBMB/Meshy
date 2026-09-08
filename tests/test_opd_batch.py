"""Teacher columns follow the Student micro-batch layout, next-token aligned."""

import pytest
import torch
from tensordict import TensorDict

from meshy.backend.titan.batch import build_micro_batch
from meshy.backend.titan.cp import CpSharder
from meshy.backend.titan.plan import MicroPlan
from meshy.backend.titan.opd import infer_top_k, layout_teacher_topk, split_teacher_topk

K = 3


class _NoCp:
    enabled = False

    def shard_seq(self, *tensors):
        return tensors


def _sample(L: int, seed: int) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 1000, (L,), generator=g)
    # Teacher row t is the distribution over token t+1: encode that by making
    # candidate 0 equal to tokens[t+1] so alignment is checkable.
    ids = torch.randint(0, 1000, (L, K), generator=g)
    ids[:-1, 0] = tokens[1:]
    return TensorDict(
        {
            "tokens": tokens,
            "logprobs": -torch.rand(L, generator=g),
            "mask_assistant": torch.ones(L),
            "advantage": torch.tensor(0.0),
            "teacher_topk_ids": ids,
            "teacher_topk_logprobs": -torch.rand(L, K, generator=g),
        },
        batch_size=[],
    )


@pytest.mark.parametrize("layout", ["padded", "packed"])
def test_teacher_rows_align_with_labels(layout):
    samples = [_sample(5, 0), _sample(8, 1), _sample(3, 2)]
    micro = MicroPlan(sample_idx=(1, 0, 2), seq_len=16 if layout == "packed" else 8, doc_lens=(8, 5, 3))
    mb = build_micro_batch(samples, micro, layout=layout, device=torch.device("cpu"),
                           sharder=_NoCp(), need_rollout_lp=False)
    ids, lps = layout_teacher_topk(samples, micro, layout=layout, device=torch.device("cpu"))
    assert ids.shape == mb.labels.shape + (K,) and lps.shape == ids.shape
    # wherever the shifted mask is on, candidate 0 is the label itself
    on = mb.mask > 0
    assert torch.equal(ids[..., 0][on], mb.labels[on])
    # and the split is the exact inverse, per sample in micro order
    back = split_teacher_topk(ids, micro, layout=layout)
    for j, idx in enumerate(micro.sample_idx):
        L = micro.doc_lens[j]
        assert torch.equal(back[j], samples[idx]["teacher_topk_ids"][:L])


def test_truncation_and_short_teacher_rows():
    samples = [_sample(6, 0)]
    micro = MicroPlan(sample_idx=(0,), seq_len=4, doc_lens=(4,))
    ids, _ = layout_teacher_topk(samples, micro, layout="padded", device=torch.device("cpu"))
    assert torch.equal(ids[0], samples[0]["teacher_topk_ids"][:4])
    short = _sample(6, 1)
    short["teacher_topk_ids"] = short["teacher_topk_ids"][:2]
    with pytest.raises(ValueError, match="Teacher seq_len"):
        layout_teacher_topk([short], micro, layout="padded", device=torch.device("cpu"))


def test_filler_micro_and_top_k_inference():
    micro = MicroPlan(sample_idx=(), seq_len=4, doc_lens=())
    ids, lps = layout_teacher_topk([], micro, layout="padded", device=torch.device("cpu"))
    assert ids.shape == (1, 4, 1) and lps.shape == (1, 4, 1)
    assert infer_top_k([_sample(2, 0)]) == K
