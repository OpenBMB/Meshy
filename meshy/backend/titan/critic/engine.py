"""Standalone PPO critic training engine (independent ForgeEngine-style replica).

A critic is *not* part of the actor trainer. It is its own SPMD replica that
owns a :class:`CriticModel` (backbone to ``norm`` + scalar ``value_head``),
its own parallelisation (TP/CP/FSDP, no pipeline), its own optimiser, and its
own training step for the value head. The only coupling to the actor is the
advantage signal, which flows through rollout sample fields — never through
shared code or shared model state.

This module keeps the engine minimal and self-contained so a single-GPU smoke
test (and later a multi-rank replica) can exercise it without the actor's
``ForgeEngine``. It mirrors ``ForgeEngine.__init__``'s shape (init_distributed →
``ParallelDims`` → meta-build → parallelize → ``to_empty`` / ``init_states`` →
optimiser) but does not require ``ModelSpec`` / ``loss`` / ``checkpointer``.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.checkpoint import checkpoint

from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.tools import utils as tools_utils
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

from .configs import critic_config_from_actor
from .model import CriticModel
from .parallel import parallelize_critic
from ..cp import CpSharder

__all__ = ["CriticEngine"]


class CriticEngine:
    """Own a value net and learn its value head (and, optionally, the backbone).

    The critic keeps its own parameters, its own discriminator against the
    actor: no parameter is shared with ``model_parts`` of the actor trainer.
    """

    def __init__(
        self,
        actor_config: "Module.Config",
        *,
        seq_len: int,
        parallelism: ParallelismConfig,
        training: TrainingConfig,
        compile: CompileConfig | None = None,
        ac_config: ActivationCheckpointConfig | None = None,
        comm: CommConfig | None = None,
        debug: DebugConfig | None = None,
        dump_folder: str = "./outputs/critic",
        lr: float = 1e-5,
        warmup_steps: int = 0,
        max_norm: float | None = None,
        train_backbone: bool = False,
        model_spec: Any = None,
    ) -> None:
        device_type = tools_utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ.get('LOCAL_RANK', 0))}")
        tools_utils.device_module.set_device(self.device)

        # Init distributed + build meshes. Use world size from env so a
        # torchrun-launched replica (like the actor) gets the real mesh.
        dist_utils.init_distributed(
            comm if comm is not None else CommConfig(),
            enable_cpu_backend=training.enable_cpu_offload,
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.parallel_dims = ParallelDims.from_config(parallelism, world_size)

        self.seq_len = seq_len
        self.training = training
        # Every batch handed to the model must have a row length divisible by
        # ``tp * cp * 2``: the head-tail CP load balancer asserts it
        # (``_HeadTailLoadBalancer._generate_indices``) and shard/restore is
        # only defined on that lattice. The actor gets this from its planner
        # (``plan.py::resolve_align``); the critic has no planner, so it pads
        # here rather than requiring every caller to remember. With CP off the
        # divisor is 1 and this is inert.
        self.seq_align = max(1, self.parallel_dims.seq_len_divisor)
        self.parallelism = parallelism
        self.compile_config = compile if compile is not None else CompileConfig()
        self.ac_config = ac_config if ac_config is not None else ActivationCheckpointConfig()
        self.debug = debug if debug is not None else DebugConfig()
        # The critic's only CP entry point: used to gather per-token values into
        # temporal order for GAE and to re-shard the result. Same load balancer
        # as the actor, so the two agree token-for-token.
        self.sharder = CpSharder(
            self.parallel_dims, parallelism.context_parallel_load_balancer
        )

        # Build the critic model on meta, then parallelize.
        critic_cfg = critic_config_from_actor(actor_config)
        # Align RoPE with the training seq length, mirroring the actor.
        critic_cfg.update_from_config(trainer_config=self._dummy_trainer_config())
        self.model_config = critic_cfg
        with (
            torch.device("meta"),
            tools_utils.set_default_dtype(
                torch.bfloat16 if training.dtype == "bfloat16" else torch.float32
            ),
        ):
            model = critic_cfg.build()

        model = parallelize_critic(
            model,
            parallel_dims=self.parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=self.compile_config,
            ac_config=self.ac_config,
            dump_folder=dump_folder,
        )

        # Mirror ``torchtitan.trainer``: under CPU offload the *parameters* are
        # created on the host, but the RoPE cache is read by the forward on the
        # accelerator, so it must be built there (``buffer_device=None`` would
        # follow the init device and leave it on CPU).
        if training.enable_cpu_offload:
            init_device, buffer_device = "cpu", torch.device(device_type)
        else:
            init_device, buffer_device = device_type, None
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_states(buffer_device=buffer_device)
        model.train()
        self.model = model
        self.model_parts = [model]
        self.model_spec = model_spec

        # Warm-start (by default) only the value head; the recipe's cold-start
        # phase trains just the critic while the actor is frozen, so backbone
        # training is opt-in.
        self.train_backbone = bool(train_backbone)
        if not self.train_backbone:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.value_head.parameters():
                p.requires_grad = True

        # Critic-only optimiser over whatever is trainable.
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)
        self.base_lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        # ``TrainingConfig.max_norm`` was previously accepted and ignored.
        self.max_norm = float(training.max_norm) if max_norm is None else max_norm
        self.train_steps = 0
        logger.info(
            "CriticEngine initialized: params={:,}, trainable={:,}, seq_len={}, "
            "world_size={}, tp={}, cp={}, lr={}, warmup={}, max_norm={}, device={}",
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in trainable),
            seq_len, world_size, parallelism.tensor_parallel_degree,
            parallelism.context_parallel_degree, self.base_lr,
            self.warmup_steps, self.max_norm, self.device,
        )

    def _dummy_trainer_config(self):
        """Adapter so ``CriticModel.Config.update_from_config`` accepts a bare config."""
        from dataclasses import dataclass

        @dataclass
        class _Training:
            seq_len: int

        @dataclass
        class _Trainer:
            training: _Training
            parallelism: ParallelismConfig

        return _Trainer(
            training=_Training(seq_len=self.seq_len),
            parallelism=self.parallelism,
        )

    @property
    def train_context(self):
        return dist_utils.get_train_context(False)

    @torch.no_grad()
    def load_backbone(self, hf_path: str) -> int:
        """Load backbone weights from an HF checkpoint, keeping the value head zeroed.

        Follows the actor's exact load path (``CheckpointManager.dcp_load``) so
        DTensor sharding under FSDP is handled:
        ``get_model_state_dict -> adapter.to_hf -> dcp.load(HF reader) ->
        adapter.from_hf -> set_model_state_dict``. The ``value_head`` keys are
        dropped before DCP and never overwritten, so the value head stays
        zero-initialised (the recipe's cold-start requirement, see
        ``CriticModel.init_states``). This is also why the critic needs no
        equivalent of miles' ``_rezero_critic_value_head``: the head never
        shares a key with anything in the checkpoint.

        Returns the number of parameters loaded.
        """
        if self.model_spec is None or self.model_spec.state_dict_adapter is None:
            raise ValueError(
                "load_backbone requires a ModelSpec with a state_dict_adapter "
                "(pass model_spec=... to CriticEngine)"
            )
        import os

        if not os.path.isdir(hf_path):
            raise FileNotFoundError(f"Not an HF checkpoint directory: {hf_path}")
        if not any(f.endswith(".safetensors") for f in os.listdir(hf_path)):
            raise FileNotFoundError(f"No *.safetensors in {hf_path}")

        from torch.distributed.checkpoint import load as dcp_load
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            set_model_state_dict,
        )

        adapter = self.model_spec.state_dict_adapter(
            self.model_spec.model, hf_path
        )
        reader = adapter.get_hf_storage_reader(hf_path)

        # Placeholder targets in *torchtitan-native* keys (DTensor on FSDP
        # meshes). Drop the value head so it is neither a target nor touched.
        native = get_model_state_dict(self.model)
        native = {k: v for k, v in native.items() if "value_head" not in k}

        # DCP needs an HF-keyed target mirror; the adapter knows the key map.
        hf_target = adapter.to_hf(native)
        dcp_load(hf_target, storage_reader=reader)

        # Back to native keys, then write into the model. Re-inject the value
        # head's current (random) values so set_model_state_dict's strict
        # matching is satisfied; they are written unchanged.
        converted = adapter.from_hf(hf_target)
        for k, v in get_model_state_dict(self.model).items():
            if "value_head" in k:
                converted[k] = v
        set_model_state_dict(self.model, converted)

        n = sum(v.numel() for v in native.values())
        logger.info("load_backbone: loaded {} param values from {}", n, hf_path)
        return n

    # ── resumable state ──────────────────────────────────────────────────
    # The critic has no ``CheckpointManager``: it is not a ForgeEngine, its
    # optimiser is a plain AdamW rather than an ``OptimizersContainer``, and it
    # has no LR scheduler (warmup is derived from ``train_steps``). DCP over
    # ``get_state_dict`` / ``set_state_dict`` covers all of that directly --
    # those helpers understand both DTensor parameters and optimizer moments
    # living on the same mesh. Losing this state is not a cosmetic failure: the
    # value net is the only thing the cold-start windows produce.

    def _dcp_state(self) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {"model": model_sd, "optim": optim_sd}

    @torch.no_grad()
    def save_state(self, path: str) -> None:
        """Write model + optimiser to ``path`` as DCP. Collective on all ranks."""
        import torch.distributed.checkpoint as dcp

        dcp.save(self._dcp_state(), checkpoint_id=path)
        logger.info("CriticEngine.save_state: wrote {} (train_steps={})", path, self.train_steps)

    @torch.no_grad()
    def load_state(self, path: str) -> None:
        """Restore model + optimiser from a DCP directory written by :meth:`save_state`.

        Mutually exclusive with :meth:`load_backbone`: the restored value head
        and backbone are the critic's learned state, and re-running the base
        model load over them would silently throw away every window of value
        training the checkpoint exists to preserve.
        """
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import set_state_dict

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Not a critic checkpoint directory: {path}")
        state = self._dcp_state()
        dcp.load(state, checkpoint_id=path)
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optim"],
        )
        logger.info("CriticEngine.load_state: restored value net + optimiser from {}", path)

    def align_from_batch(self, batch: dict[str, torch.Tensor]) -> int:
        """Padded width ``make_batch`` must use for this batch to survive CP.

        ``make_batch`` pads only to the longest row in the batch, which is not
        generally a multiple of the CP divisor -- and the head-tail load
        balancer asserts that it is. Returning a rounded-up ``pad_to`` here
        keeps the padding decision in the engine that knows the parallel
        layout, rather than in each caller.
        """
        rows = batch["input_ids"]
        longest = int(rows.shape[1])
        aligned = -(-longest // self.seq_align) * self.seq_align
        return min(aligned, self.seq_len)

    def make_batch(
        self,
        samples: list[Any],
        *,
        device: Any = None,
        shift: bool = True,
        pad_to: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """``data.make_batch`` with the CP-safe padding width applied.

        Prefer this over calling ``data.make_batch`` directly: the raw helper
        pads to the longest row, which trips the CP load balancer's divisibility
        assertion for any batch whose longest row is not a multiple of
        ``tp * cp * 2`` -- including every real batch on an odd token count.

        ``shift`` defaults to the pipeline's next-token grid (see
        ``data.make_batch``); pass ``shift=False`` only for inspection.

        ``pad_to`` overrides the derived width for callers that already decided
        it -- the dynamic-batching planner picks a micro-batch's row length
        when it packs the micro-batch, and it must be the width actually used
        or the plan's token accounting and the batch disagree. It still has to
        sit on the CP lattice, so it is validated rather than trusted.
        """
        from .data import make_batch

        device = self.device if device is None else device
        if pad_to is None:
            longest = max((len(s.tokens) for s in samples), default=0)
            # Round the padded width up to the CP lattice. ``seq_align`` is 1
            # when CP (and TP) are off, so this degenerates to the raw helper.
            pad_to = -(-longest // self.seq_align) * self.seq_align
        elif pad_to % self.seq_align != 0:
            raise ValueError(
                f"pad_to ({pad_to}) must be a multiple of the CP/TP sequence "
                f"divisor ({self.seq_align})"
            )
        return make_batch(
            samples, device=device, pad_to=min(pad_to, self.seq_len), shift=shift
        )

    @torch.no_grad()
    def hidden_states(
        self,
        input_ids: torch.Tensor,   # [rows, S] full sequence
        positions: torch.Tensor,   # [rows, S]
        attention_masks: Any = None,
    ) -> torch.Tensor:
        """Post-``norm`` hidden states ``[rows, S, D]`` -- the value head's input.

        The value head is a linear map on exactly this tensor, so exposing it
        makes the value loss and its gradient checkable in closed form. Used by
        ``scripts/smoke/critic_engine_cp.py`` to verify the CP reduction
        independently of the engine's own arithmetic.
        """
        l_input, l_pos = self.sharder.shard_seq(input_ids, positions)
        with self.train_context():
            _, hidden = self.model(
                l_input, positions=l_pos, attention_masks=attention_masks, return_hidden=True
            )
        # ``gather_seq`` permutes along dim 1 of a 2-D tensor, so the feature
        # dim has to be folded into the *batch* dim by permuting first -- a
        # plain ``reshape(rows * dim, s_local)`` interleaves feature and
        # sequence strides and silently returns the wrong values.
        rows, s_local, dim = hidden.shape
        flat = hidden.permute(0, 2, 1).reshape(rows * dim, s_local).contiguous()
        (gathered,) = self.sharder.gather_seq(flat)
        return gathered.reshape(rows, dim, -1).permute(0, 2, 1)

    @torch.no_grad()
    def predict_values(
        self,
        input_ids: torch.Tensor,   # [rows, S] full sequence
        positions: torch.Tensor,   # [rows, S]
        attention_masks: Any = None,
    ) -> torch.Tensor:
        """Forward pass → per-token values ``[rows, S]`` (no grad).

        Takes and returns **full** sequences. Context parallelism is handled
        inside: the input is sharded for the model (which is what
        ``apply_cp_to_forward`` expects -- the actor does the same in
        ``batch.py::build_micro_batch``) and the per-token output is gathered
        back into temporal order. Callers never see a CP-local tensor, so they
        cannot accidentally mix layouts.
        """
        l_input, l_pos = self.sharder.shard_seq(input_ids, positions)
        with self.train_context():
            values = self.model(l_input, positions=l_pos, attention_masks=attention_masks)
        (full,) = self.sharder.gather_seq(values.squeeze(-1))
        return full

    @torch.no_grad()
    def predict_vapo_gae(
        self,
        input_ids: torch.Tensor,     # [rows, S] full sequence
        positions: torch.Tensor,     # [rows, S]
        rewards: torch.Tensor,       # [rows, S] per-token reward
        mask: torch.Tensor,          # [rows, S] float assist-mask
        doc_ids: torch.Tensor,       # [rows, S]
        n_docs: int,
        attention_masks: Any = None,
        *,
        gamma: float = 1.0,
        alpha: float = 1.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the critic forward, then VAPO-GAE to per-sequence advantages.

        Every tensor in and out is a **full** ``[rows, S]`` sequence; CP lives
        inside :meth:`predict_values`. That matters here specifically: GAE is a
        backward recursion over *adjacent* timesteps, and the head-tail CP
        layout deliberately makes neighbouring positions non-adjacent, so the
        recursion is only meaningful on the gathered sequence.

        Returns ``(per_token_adv, per_seq_adv, values)``:

        * ``per_token_adv`` ``[rows, S]`` — GAE advantages, masked to ``mask``;
        * ``per_seq_adv`` ``[n_docs]`` — the masked token-mean per sample,
          exactly the shape :class:`meshy.backend.titan.Batch.advantages`
          expects, and identical on every CP rank;
        * ``values`` ``[rows, S]`` — the per-token values the advantages were
          built from, returned so a caller computing health gauges does not
          have to pay for a second forward.

        Lambda is per-document via the recipe's VAPO rule λᵢ=1−1/(α·Lᵢ), with
        Lᵢ the document's masked response-token count.
        """
        from .gae import advantages_to_per_sequence, compute_vapo_gae

        values = self.predict_values(
            input_ids, positions, attention_masks=attention_masks
        )  # [rows, S], already gathered
        adv, _ = compute_vapo_gae(
            rewards, values, mask, doc_ids, n_docs, gamma=gamma, alpha=alpha
        )
        adv = adv * mask
        per_seq = advantages_to_per_sequence(adv, mask, doc_ids, n_docs)
        return adv, per_seq, values

    def train_value(
        self,
        input_ids: torch.Tensor,   # [rows, S] full sequence
        positions: torch.Tensor,   # [rows, S]
        targets: torch.Tensor,     # [rows, S] per-token value targets (returns)
        mask: torch.Tensor,        # [rows, S] float assist-mask
        attention_masks: Any = None,
    ) -> float:
        """One value-loss backward + clipped optimiser step.

        Increments :attr:`train_steps` and applies the lr warmup, so the
        warmup is a property of the critic rather than of whichever caller
        drives it. Returns the masked mean value loss (detached, for logging).

        Takes **full** ``[rows, S]`` tensors and shards them for the model
        internally. The loss is pointwise in the sequence dimension, so unlike
        GAE it needs no gather -- but it does need the loss-mesh reduction:
        FSDP's mesh spans CP (``parallel_dims.fsdp = dp_shard * cp``) with
        gradient division disabled, so every rank's local gradient is *summed*.
        Each rank therefore contributes its own shard's numerator while the
        denominator is the global mask count across DP and CP, which makes the
        summed gradient equal the gradient of the global mean.
        """
        loss, _ = self.train_value_accumulated(
            [(input_ids, positions, targets, mask, attention_masks)]
        )
        return loss

    def train_value_accumulated(
        self,
        batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]],
        *,
        collect_values: bool = False,
    ) -> tuple[float, list[torch.Tensor]]:
        """Accumulate value gradients over several micro-batches.

        ``batches`` are ``(input_ids, positions, targets, mask, attention_masks)``
        tuples. One optimizer update is performed after all forwards/backwards.
        The denominator is the total assistant-token count across every
        micro-batch and every data/CP rank, so this is mathematically the same
        as fitting one concatenated mini-batch while keeping peak memory at the
        micro-batch size.

        Returns ``(loss, values)``. With ``collect_values`` the second element
        is one gathered ``[rows, S]`` CPU tensor per micro-batch, in ``batches``
        order; otherwise it is empty.

        Those values are worth taking: **every forward here runs on the
        pre-update weights**, because the optimizer steps only once, after the
        whole list has contributed gradients. So they are exactly what a
        separate no-grad scoring pass over the same rows would return -- and
        that pass is a fifth of the critic's compute (one forward against the
        forward + recompute + backward this does). A caller that needs the
        pre-update value function for publishing or for the health gauges
        should take it from here rather than pay for it twice. If the caller
        loops for several value epochs, only the first call's values carry that
        guarantee; the later ones see weights this step already moved.
        """
        if not batches:
            raise ValueError("train_value_accumulated requires at least one batch")

        self.train_steps += 1
        self._apply_warmup()
        self.optimizer.zero_grad()

        # Count the CP-local masks, not the original full masks: every CP rank
        # receives a different sequence shard, so summing full masks would
        # multiply the denominator by the CP degree.
        local_count = torch.zeros((), device=self.device, dtype=torch.float32)
        for input_ids, positions, targets, mask, _ in batches:
            input_ids = input_ids.to(self.device, non_blocking=True)
            positions = positions.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            _, _, _, local_mask = self.sharder.shard_seq(
                input_ids, positions, targets, mask
            )
            local_count = local_count + local_mask.float().sum()
        denom = self._global_sum(local_count).clamp(min=1.0)
        local_numerator = torch.zeros((), device=self.device, dtype=torch.float32)
        collected: list[torch.Tensor] = []

        for input_ids, positions, targets, mask, attention_masks in batches:
            # Offline callers may keep the queued micro-batches on CPU so a
            # large effective mini-batch does not occupy GPU memory between
            # backward passes.
            input_ids = input_ids.to(self.device, non_blocking=True)
            positions = positions.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            l_input, l_pos, l_targets, l_mask = self.sharder.shard_seq(
                input_ids, positions, targets, mask
            )
            with self.train_context():
                values = self.model(
                    l_input, positions=l_pos, attention_masks=attention_masks
                ).squeeze(-1)
                numerator = ((values - l_targets) ** 2 * l_mask).sum()
                loss = numerator / denom
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite critic loss: {loss.item()}")
                local_numerator = local_numerator + numerator.detach().float()
                loss.backward()
            if collect_values:
                # Back to temporal order before it leaves the engine: the CP
                # shard is a head-tail permutation, so a caller slicing row
                # ``j`` to its own length would otherwise get the wrong tokens.
                (full,) = self.sharder.gather_seq(values.detach())
                collected.append(full.float().cpu())

        # The cold-start value-loss spike is large (recipe §2 reports a peak
        # around 32); an unclipped step on it moves the head a long way.
        self.clip_grad_norm()
        self.optimizer.step()

        global_numerator = self._global_sum(local_numerator)
        return float((global_numerator / denom).detach()), collected

    def _global_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Sum a scalar over the loss mesh (DP + CP), preserving gradients off."""
        mesh = self.parallel_dims.get_optional_mesh("loss")
        if mesh is None:
            return value
        out = value.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=mesh.get_group())
        return out

    def _apply_warmup(self) -> None:
        """Linear lr warmup over the first ``warmup_steps`` value steps."""
        if self.warmup_steps <= 0:
            return
        scale = min(1.0, self.train_steps / self.warmup_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * scale

    def clip_grad_norm(self) -> float | None:
        """Clip trainable gradients to ``max_norm``; ``None`` when disabled."""
        if self.max_norm is None or self.max_norm <= 0:
            return None
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            return None
        # DTensor-aware: torchtitan's helper handles the FSDP/TP sharded case,
        # where a per-rank norm would be wrong.
        total = dist_utils.clip_grad_norm_(
            trainable,
            self.max_norm,
            foreach=True,
            pp_mesh=None,
        )
        return float(total)


__all__ = ["CriticEngine"]
