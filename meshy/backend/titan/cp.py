"""Single-class wrapper around torchtitan's context-parallel primitives.

:class:`CpSharder` is the only place the trainer talks to CP:

* :meth:`shard_seq` — slice a batch of ``[B, S]`` per-token tensors into the
  CP-local ``[B, S_local]`` head-tail layout (no-op when CP is disabled).
* :meth:`all_reduce_sum` — a gradient-free SUM over the CP group, used for
  the per-sequence token counts and for metrics.

Loss reductions deliberately do **not** all-reduce the differentiable
numerator across CP: every CP rank backpropagates its local partial sum
divided by the *global* denominator, and FSDP's gradient reduce (SUM over
``dp_shard * cp``; torchtitan disables the automatic division) assembles the
global gradient. That is exactly the same contract as data parallelism.

Determinism note
----------------
Every call to :meth:`shard_seq` constructs a fresh ``_HeadTailLoadBalancer``
under the same ``(seq_len, cp_world_size)``, which yields a deterministic
permutation. As long as all per-token tensors of a micro-batch are sharded
through one call they are token-for-token aligned on every CP rank — which
is what makes ``new_lp - old_lp`` meaningful under CP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


class CpSharder:
    def __init__(
        self,
        parallel_dims: "ParallelDims",
        load_balancer: str | None = "headtail",
    ) -> None:
        self._parallel_dims = parallel_dims
        self._load_balancer = load_balancer

    @property
    def enabled(self) -> bool:
        return self._parallel_dims.cp_enabled

    @property
    def load_balancer(self) -> str | None:
        return self._load_balancer

    def shard_seq(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Shard each tensor along ``seq_dim=1`` across the CP mesh.

        All tensors are sharded together with a single
        ``_context_parallel_shard`` call, so they receive identical head-tail
        permutations and end up token-for-token aligned on every CP rank.

        No-op (returns the inputs unchanged) when CP is disabled.
        """
        if not self.enabled:
            return tensors

        from torchtitan.distributed.context_parallel import cp_shard

        cp_mesh = self._parallel_dims.get_mesh("cp")
        sharded, _ = cp_shard(
            cp_mesh,
            tuple(tensors),
            None,
            self._load_balancer,
            input_seq_dim=1,
        )
        return sharded

    @torch.no_grad()
    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """SUM ``value`` over the CP group (no autograd). Identity when CP is off."""
        if not self.enabled:
            return value
        out = value.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self._parallel_dims.get_mesh("cp").get_group())
        return out
