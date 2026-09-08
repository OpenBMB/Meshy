# Copyright (c) Meshy project. Adapted from
# torchtitan/models/llama3/state_dict_adapter.py (BSD-style license).
"""HF<->torchtitan state-dict adapter for MiniCPM5.

MiniCPM5 dense checkpoints use a Llama layout with ``n_kv_heads=2``. Upstream
``Llama3StateDictAdapter._permute`` does ``w.view(n_heads, ...)`` directly
on FSDP-sharded DTensors. When the FSDP mesh is larger than ``n_kv_heads``
(e.g. 4 GPUs), DTensor refuses the view:

    Cannot unflatten unevenly sharded tensor: output dimension 0 (size 2)
    is not evenly divisible by mesh dimension 0 (size 4).
    Please redistribute the tensor before this operation.

Checkpoint load hits this in ``to_hf`` (container reshape before DCP)
and again in ``from_hf`` (RoPE reverse-permute after DCP). Gather to a
local tensor, run the original permute, then re-shard with the same
placements so DCP still sees the FSDP layout.

The upstream adapter also assumes ``head_dim == dim // n_heads`` when it
sizes the K permute, which does not hold for MiniCPM5-1B (dim=1536,
head_dim=128); see ``_permute`` below.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.distributed.tensor import DTensor, distribute_tensor
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter


class MiniCPM5StateDictAdapter(Llama3StateDictAdapter):
    """Llama3 adapter with FSDP/DTensor-safe RoPE permute."""

    # ``dim1``/``dim2`` are dropped on purpose: Llama3 computes them from
    # ``head_dim = dim // n_heads``, which is 96 for MiniCPM5-1B (dim=1536)
    # while the checkpoint uses head_dim=128. Leaving them ``None`` makes the
    # upstream permute read the real K width from the tensor itself.
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        return self._on_local(
            w,
            lambda local: super(MiniCPM5StateDictAdapter, self)._permute(
                local, n_heads_arg
            ),
        )

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        return self._on_local(
            w,
            lambda local: super(MiniCPM5StateDictAdapter, self)._reverse_permute(
                local, n_heads_arg
            ),
        )

    @staticmethod
    def _on_local(w: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]):
        if not isinstance(w, DTensor):
            return fn(w)
        out = fn(w.full_tensor())
        return distribute_tensor(out, w.device_mesh, list(w.placements))
