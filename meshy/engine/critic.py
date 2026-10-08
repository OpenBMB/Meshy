"""SPMD wrapper around :class:`CriticEngine` — one replica, one command loop.

:class:`CriticEngine` is a bare compute object (model + optimiser + value
step). This module gives it the same replica shape the actor trainer has:
rank 0 originates work, the window is broadcast so every rank executes the same
collectives, and only rank 0 returns a result. It is the critic's counterpart
to :class:`meshy.engine.opd.OPDTeacherEngine`.

The `CriticEngine` is built in :meth:`setup`, not ``__init__``, so
:meth:`SpmdEngine._init_process_group` has already created the process group by
the time it runs. ``torchtitan``'s ``init_distributed`` is idempotent (it warns
and returns when a group already exists), so the inner engine attaches to that
group rather than trying to build a second one.
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from meshy.engine.spmd import SpmdEngine


class CriticSpmdEngine(SpmdEngine):
    """A critic value net driven through the replica command loop."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_device_id: str,
        master_addr: str,
        master_port: int,
        runtime_root: str,
        name: str,
        model_path: str,
        trainer_config: Any,
        train_backbone: bool = True,
        lr: float = 5e-6,
        warmup_steps: int = 10,
        optimizer_kwargs: dict | None = None,
        value_clip: float | None = None,
        value_epochs: int = 1,
        gamma: float = 1.0,
        alpha: float = 1.5,
        max_norm: float = 1.0,
        micro_rows: int = 1,
        max_tokens_per_micro: int | None = None,
        publish_mode: str = "values",
        is_colocate: bool = False,
        checkpoint: Any = None,
        resume_step: int = 0,
        bind_host: str = "127.0.0.1",
        endpoint_port: int = 0,
    ) -> None:
        super().__init__(
            rank=rank,
            world_size=world_size,
            local_device_id=local_device_id,
            master_addr=master_addr,
            master_port=master_port,
            bind_host=bind_host,
            endpoint_port=endpoint_port,
            runtime_root=runtime_root,
            name=name,
        )
        self.model_path = model_path
        self.trainer_config = trainer_config
        self.train_backbone = bool(train_backbone)
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.optimizer_kwargs = optimizer_kwargs
        self.value_clip = value_clip
        self.value_epochs = int(value_epochs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.max_norm = float(max_norm)
        #: rows per critic forward, used only when ``max_tokens_per_micro`` is
        #: unset; long sequences need this small
        self.micro_rows = int(micro_rows)
        #: Token budget per critic forward. Preferred over :attr:`micro_rows`:
        #: a window's rows span two orders of magnitude in length (p50 ~10k,
        #: p90 ~40k at 128k), so a fixed row count either wastes the batch
        #: dimension on the short rows or blows memory on the long ones. With a
        #: budget the planner sorts by length and packs to it, which is what
        #: collapses a 480-row window from 480 forwards into a few dozen.
        self.max_tokens_per_micro = (
            int(max_tokens_per_micro) if max_tokens_per_micro else None
        )
        if publish_mode not in ("values", "advantage"):
            raise ValueError(
                f"publish_mode must be 'values' or 'advantage', got {publish_mode!r}"
            )
        #: ``"values"`` writes the per-token value function back to the rollout
        #: row and leaves GAE to the trainer (``enable_gae``); ``"advantage"``
        #: is the original arrangement where the critic reduces to one scalar
        #: per sequence. See ``docs/critic_engine.md``.
        self.publish_mode = publish_mode
        self.is_colocate = bool(is_colocate)
        self.critic: Any = None
        #: set by rank 0 in ``score_and_train`` just before the command is
        #: broadcast; other ranks receive it over the wire.
        self._publish = True
        #: the actor weight version these advantages were produced against;
        #: reported on the cold-start gate so the rollout keeps generating
        #: against the weights it already holds.
        self.weight_version = 0

        from meshy.config import GlobalCheckpointConfig

        self.checkpoint = GlobalCheckpointConfig.coerce(checkpoint)
        #: Global window index to restore to; 0 for a fresh run.
        self.resume_step = int(resume_step)
        #: Set by rank 0 in ``score_and_train`` and broadcast with the window,
        #: like ``_publish``. The critic's window index *is* the run's global
        #: clock (it advances from the very first window, cold start
        #: included), but the counter itself belongs to the Worker.
        self._global_step = 0
        self._store = None

    # ── SpmdEngine hooks ────────────────────────────────────────────────
    def setup(self) -> None:
        from torchtitan.config.configs import (
            ActivationCheckpointConfig,
            CompileConfig,
            DebugConfig,
            ParallelismConfig,
            TrainingConfig,
        )

        from meshy.backend.titan.critic.engine import CriticEngine
        from meshy.backend.titan.config import _get_model_spec

        cfg = self.trainer_config
        spec = _get_model_spec(
            cfg.model_name, cfg.model_flavor, getattr(cfg, "attn_backend", "sdpa")
        )
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=cfg.dp_shard_degree,
            data_parallel_replicate_degree=cfg.dp_replicate_degree,
            tensor_parallel_degree=cfg.tp_degree,
            context_parallel_degree=cfg.cp_degree,
        )
        training = TrainingConfig(
            seq_len=cfg.seq_len,
            local_batch_size=self.micro_rows,
            global_batch_size=self.micro_rows * self.world_size,
            max_norm=self.max_norm,
            steps=cfg.steps,
            dtype=cfg.dtype,
        )
        self.critic = CriticEngine(
            spec.model,
            seq_len=cfg.seq_len,
            parallelism=parallelism,
            training=training,
            compile=CompileConfig(enable=False, components=[]),
            # Long-context critics need real activation checkpointing: the
            # retained per-layer activations scale with the CP-local sequence
            # length, and with train_backbone the backward keeps them all.
            ac_config=ActivationCheckpointConfig(
                mode=getattr(cfg, "activation_checkpoint_mode", "selective")
            ),
            debug=DebugConfig(),
            dump_folder=cfg.dump_folder,
            lr=self.lr,
            warmup_steps=self.warmup_steps,
            optimizer_kwargs=self.optimizer_kwargs,
            value_clip=self.value_clip,
            max_norm=self.max_norm,
            train_backbone=self.train_backbone,
            model_spec=spec,
        )
        if not self._restore_state():
            # Cold start (recipe §2): backbone from the base model, value head
            # random. Never the actor's current weights -- the critic must not
            # inherit the policy's drift.
            n = self.critic.load_backbone(self.model_path)
            logger.info(
                "Critic {} rank {}: loaded {:,} backbone params from {}; value head random",
                self.name, self.rank, n, self.model_path,
            )
        if self.is_colocate:
            self.offload_to_cpu()

    def execute(self, payload: dict[str, Any], samples: list[Any] | None) -> Any:
        action = payload.get("action")
        if action == "score_and_train":
            return self._score_and_train_impl(samples)
        # GPU residency is broadcast like any other command so every rank moves
        # together; the colocation manager only runs on rank 0.
        if action == "colocate_acquire":
            self.restore_to_gpu()
            return None
        if action == "colocate_release":
            self.offload_to_cpu()
            return None
        raise ValueError(f"unknown critic command {action!r}")

    # ── global checkpoint ────────────────────────────────────────────────
    @property
    def store(self):
        """Lazily-built :class:`~meshy.service.checkpoint.StateStore`."""
        if self._store is None:
            self._store = self.runtime.state_store()
        return self._store

    def _restore_state(self) -> bool:
        """Reload the value net and optimiser; ``False`` means start cold.

        A resume *inside* the cold start is the case worth getting right: the
        actor has never stepped and has nothing to restore, but the critic has
        20-30 windows of value training that would otherwise be redone.
        """
        from meshy.service.checkpoint import log_resume

        if self.resume_step <= 0:
            return False
        path = self.store.shard_dir(self.name, self.resume_step)
        payload = self.store.payload(self.name, self.resume_step)
        self.critic.load_state(path)
        self.critic.train_steps = int(payload.get("train_steps", 0))
        if self.is_master:
            log_resume(
                self.name,
                self.resume_step,
                f"train_steps={self.critic.train_steps} (LR warmup position preserved)",
            )
        return True

    def _save_global_checkpoint(self, global_step: int) -> None:
        """Persist the critic's shard. Collective: every rank must call."""
        path = self.store.shard_dir(self.name, global_step)
        self.critic.save_state(path)
        self._barrier()
        if not self.is_master:
            return
        self.store.write_meta(
            self.name,
            global_step,
            {
                "train_steps": int(self.critic.train_steps),
                "windows": int(global_step),
                "weight_version": int(self.weight_version),
            },
        )
        self.store.mark_complete(self.name, global_step)
        self.store.prune(self.name, self.checkpoint.keep_latest_k)
        logger.info(
            "Critic {} checkpointed global step {} -> {}", self.name, global_step, path
        )

    # ── public API (rank 0) ─────────────────────────────────────────────
    def score_and_train(
        self, samples: list[Any], *, publish: bool = True, global_step: int = 0
    ) -> Any:
        """Score one window, publish the result, then update the value net.

        What is published depends on :attr:`publish_mode`: ``"values"`` returns
        the per-token value function per row (the trainer turns it into GAE
        advantages), ``"advantage"`` returns one reduced scalar per sequence.

        ``publish=False`` during the critic's cold start: the window still
        trains the value net, but nothing is returned, so the actor sees
        nothing (recipe §2).

        ``global_step`` is the window index this call completes, on the run's
        shared clock; it decides whether this window ends on a checkpoint.
        """
        # Read by rank 0 only, immediately before the broadcast inside
        # ``_score_and_train_impl`` puts them on the wire for every rank.
        self._publish = bool(publish)
        self._global_step = int(global_step)
        return self.submit_command("score_and_train", samples=samples)

    # ── implementation (all ranks) ──────────────────────────────────────
    def _score_and_train_impl(self, samples: list[Any] | None) -> dict[str, Any] | None:
        """One window: plan it, fit the value net, ship the result to rank 0.

        The window is *sharded* across the DP mesh and *packed* by token budget
        inside each shard, both by the same planner the actor trainer uses
        (``meshy.backend.titan.plan``). Before that it was neither: every rank
        ran every row, so ``dp_shard`` bought memory and no throughput, and
        ``micro_rows`` rows per forward meant a 480-row window was 480
        forwards whatever the rows were made of.

        There is also no separate scoring pass any more. The value step is a
        single optimiser update over gradient-accumulated micro-batches, so
        every forward inside it already runs on the pre-update weights -- the
        same weights a scoring pass would have used, over the same rows. The
        values come back out of that forward instead of being recomputed, so
        "score with the pre-update V, publish, then update it" still holds
        exactly, at one forward per row rather than two.
        """
        from meshy.backend.titan.critic.data import samples_from_rows
        from meshy.backend.titan.critic.gae import (
            advantages_to_per_sequence,
            compute_returns,
            compute_vapo_gae,
        )
        from meshy.backend.titan.critic.metrics import (
            Accumulator,
            diagnostics_from_states,
        )

        if self.critic is None:
            raise RuntimeError("CriticSpmdEngine.score_and_train() called before init()")

        broadcast = self._broadcast(
            {
                "samples": samples,
                "publish": self._publish,
                "global_step": self._global_step,
            }
            if self.is_master
            else None
        )
        rows = broadcast["samples"]
        publish = bool(broadcast["publish"])
        global_step = int(broadcast["global_step"])
        batch_samples = samples_from_rows(rows)

        # ---- 1. plan: this rank's rows, packed into micro-batches --------
        plan, dp_rank = self._plan_window(batch_samples)
        local = plan.local_samples(batch_samples, dp_rank)
        local_indices = plan.local_indices[dp_rank]
        # ``mini_batch_size`` is the whole local shard, so there is exactly one
        # mini-batch -- one optimiser step per window, which is the clock
        # recipe §2 counts in ("10 iter lr warmup" = 10 rollout steps).
        micros = plan.per_rank[dp_rank][0].micros

        micro_batches: list[tuple[Any, ...]] = []
        built: list[tuple[Any, dict[str, torch.Tensor], torch.Tensor]] = []
        for m in micros:
            chunk = [local[p] for p in m.sample_idx]
            # Filler micro-batches keep the number of forwards equal across
            # ranks; FSDP issues collectives per forward/backward, so a rank
            # with fewer would hang the others. They carry an all-zero mask and
            # contribute exactly zero to both the loss and its denominator.
            b = (
                self.critic.make_batch(chunk, device="cpu", pad_to=m.seq_len)
                if chunk
                else self._filler_batch(m.seq_len)
            )
            targets = compute_returns(
                b["rewards"], b["mask"], b["doc_ids"], gamma=self.gamma
            )
            micro_batches.append(
                (b["input_ids"], b["positions"], targets, b["mask"], None)
            )
            built.append((m, b, targets))

        # ---- 2. fit the value net, keeping the pre-update forward --------
        # Queued micro-batches stay on CPU; the engine moves one at a time to
        # the device immediately before its forward. Only the first epoch's
        # values are the pre-update ones -- a later epoch sees weights this
        # window already moved.
        losses: list[float] = []
        window_values: list[torch.Tensor] = []
        for epoch in range(max(1, self.value_epochs)):
            loss, collected = self.critic.train_value_accumulated(
                micro_batches, collect_values=(epoch == 0),
                old_values=window_values if epoch > 0 and self.value_clip is not None else None,
            )
            losses.append(loss)
            if epoch == 0:
                window_values = collected

        # ---- 3. gauges and the published column, per local row -----------
        if len(window_values) != len(built):
            # ``zip`` would truncate silently, and a window short of rows only
            # surfaces later as a wrong gauge or a missing published column.
            raise RuntimeError(
                f"value step returned {len(window_values)} micro-batches of "
                f"values for {len(built)} micro-batches"
            )
        accumulator = Accumulator()
        published: dict[int, Any] = {}
        for (m, b, targets), v in zip(built, window_values):
            if m.is_filler:
                continue
            accumulator.update(
                v, targets, b["mask"], b["row_rewards"], b["group_ids"]
            )
            if not publish:
                continue
            if self.publish_mode == "advantage":
                adv, _ = compute_vapo_gae(
                    b["rewards"], v, b["mask"], b["doc_ids"], len(m.sample_idx),
                    gamma=self.gamma, alpha=self.alpha,
                )
                per_seq = advantages_to_per_sequence(
                    adv * b["mask"], b["mask"], b["doc_ids"], len(m.sample_idx)
                )
                for j, p in enumerate(m.sample_idx):
                    published[local_indices[p]] = float(per_seq[j])
            else:
                # Trimmed to the row's own length: the padded tail is not part
                # of the sample and the trainer slices by ``L``.
                for j, p in enumerate(m.sample_idx):
                    published[local_indices[p]] = v[j, : len(local[p].tokens)].clone()

        # ---- 4. checkpoint, while the value net is still resident ---------
        # Every rank is inside the same broadcast command and (when colocated)
        # the critic still holds the GPU token, so the collective DCP write is
        # both safe and free of a CPU->GPU round trip.
        if self.checkpoint.due(global_step):
            self._save_global_checkpoint(global_step)

        # ---- 5. back to rank 0 -------------------------------------------
        gathered = self._gather_to_master(
            {"diagnostics": accumulator.state, "published": published}
            if self._dp_representative
            else None
        )
        if not self.is_master:
            return None

        result: dict[str, Any] = {
            "value_loss": sum(losses) / max(1, len(losses)),
            "rows": len(batch_samples),
            "diagnostics": diagnostics_from_states(
                [p["diagnostics"] for p in gathered if p is not None]
            ),
        }
        if publish:
            from tensordict import TensorDict

            by_index: dict[int, Any] = {}
            for part in gathered:
                if part is not None:
                    by_index.update(part["published"])
            if len(by_index) != len(batch_samples):
                raise RuntimeError(
                    f"critic published {len(by_index)} rows for "
                    f"{len(batch_samples)} input rows; the DP plan lost some"
                )
            ordered = [by_index[i] for i in range(len(batch_samples))]
            if self.publish_mode == "values":
                result["values"] = [
                    TensorDict({"values": v}, batch_size=[]) for v in ordered
                ]
            else:
                result["advantage"] = [
                    TensorDict(
                        {"advantage": torch.tensor(float(a), dtype=torch.float32)},
                        batch_size=[],
                    )
                    for a in ordered
                ]
        return result

    # ── window planning / result collection ─────────────────────────────
    def _plan_window(self, batch_samples: list[Any]) -> tuple[Any, int]:
        """The DP split and micro-batch packing for one window.

        Every rank runs this on the same broadcast row list and therefore
        derives the same plan without communicating it -- the same contract
        ``meshy.backend.titan.parallel.split_batch_to_local`` relies on. The
        ``Plan`` itself is kept (rather than going through that helper) because
        the critic needs ``local_indices`` to put each row's value back under
        its original position in the window.
        """
        from meshy.backend.titan.parallel import dp_rank_and_size
        from meshy.backend.titan.plan import PlannerConfig, build_plan

        dp_rank, dp_size = dp_rank_and_size(self.critic.parallel_dims)
        lengths = [len(s.tokens) for s in batch_samples]
        # The value loss lives on the shifted assistant mask; the shift never
        # drops a response token (a response never starts at index 0), so the
        # row's response-token count is that denominator.
        loss_tokens = [s.n_response for s in batch_samples]
        cfg = PlannerConfig(
            # ``padded`` keeps plain causal attention, which is what
            # torchtitan's ring-attention CP needs; ``packed`` (varlen) is not
            # available under CP.
            layout="padded",
            # One optimiser step per window: the mini-batch is the whole local
            # shard.
            mini_batch_size=max(1, len(batch_samples) // max(1, dp_size)),
            seq_len=self.critic.seq_len,
            align=self.critic.seq_align,
            max_tokens_per_micro=self.max_tokens_per_micro,
            micro_batch_size=None if self.max_tokens_per_micro else self.micro_rows,
        )
        return build_plan(lengths, loss_tokens, dp_size, cfg), dp_rank

    def _filler_batch(self, seq_len: int) -> dict[str, torch.Tensor]:
        """A one-row zero batch: same collectives, no contribution to the loss."""
        zeros_f = torch.zeros(1, seq_len, dtype=torch.float32)
        return {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "positions": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            "mask": zeros_f,
            "doc_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "rewards": zeros_f.clone(),
            "row_rewards": torch.zeros(1, dtype=torch.float32),
            "group_ids": torch.zeros(1, dtype=torch.long),
        }

    @property
    def _dp_representative(self) -> bool:
        """Whether this rank speaks for its DP shard when results go to rank 0.

        Ranks inside a CP (or TP) group all hold the same rows, and
        ``gather_seq`` has already given them the same values, so having every
        one of them ship its copy would multiply the gathered payload by the CP
        degree for nothing.
        """
        dims = self.critic.parallel_dims
        if dims.cp_enabled and dims.get_mesh("cp").get_local_rank() != 0:
            return False
        if dims.tp_enabled and dims.get_mesh("tp").get_local_rank() != 0:
            return False
        return True

    def _gather_to_master(self, payload: Any) -> list[Any]:
        """Collect one payload per rank on rank 0; ``[]`` elsewhere."""
        if self.world_size == 1:
            return [payload]
        import torch.distributed as dist

        out: list[Any] | None = [None] * self.world_size if self.is_master else None
        dist.gather_object(payload, out, dst=0, group=self.group_gloo)
        return out if out is not None else []

    # ── colocation ──────────────────────────────────────────────────────
    def offload_to_cpu(self) -> None:
        if self.critic is None:
            return
        self.critic.model.to("cpu")
        # ``model.to`` does not reach the optimizer: AdamW keeps two fp32
        # tensors per trainable parameter in ``optimizer.state``, keyed by the
        # parameter object. Left behind they hold ~2x the sharded parameter
        # bytes on the card for the rest of the run -- see
        # ``TitanTrainer._move_optimizer_states``.
        self._move_optimizer_states("cpu")
        self._sync_rope_cache()
        torch.cuda.empty_cache()

    def restore_to_gpu(self) -> None:
        if self.critic is None:
            return
        self.critic.model.to(self.critic.device)
        self._move_optimizer_states(self.critic.device)
        self._sync_rope_cache()

    def _sync_rope_cache(self) -> None:
        """Re-point ``rope.cache`` at the ``freqs_cis`` buffer after a move.

        ``RoPE.cache`` is a plain attribute, not a registered buffer, so
        ``model.to(...)`` rebinds ``freqs_cis`` (the tensor the forward reads)
        and leaves ``cache`` referencing the old device's storage -- keeping a
        full RoPE cache alive on the card across every hand-off. Same fixup as
        ``TitanTrainer.restore_to_gpu``.
        """
        model = getattr(self.critic, "model", None)
        rope = getattr(model, "rope", None)
        if rope is not None and torch.is_tensor(getattr(model, "freqs_cis", None)):
            rope.cache = model.freqs_cis

    def _move_optimizer_states(self, device: Any) -> None:
        optimizer = getattr(self.critic, "optimizer", None)
        if optimizer is None:
            return
        target = torch.device(device)
        for state in optimizer.state.values():
            for key, value in state.items():
                # ``step`` is a scalar whose device belongs to the AdamW
                # variant in use (fused/capturable); moving it breaks those
                # kernels' assumptions.
                if key == "step" or not torch.is_tensor(value):
                    continue
                if value.device.type != target.type:
                    state[key] = value.to(target)

    def on_colocate_acquire(self, grant: Any) -> None:
        del grant
        # The colocation manager runs on rank 0 only, so the move has to go
        # through the command broadcast or the other ranks stay on CPU while
        # their inputs arrive on the GPU.
        if self.world_size > 1 and self.is_master:
            self.submit_command("colocate_acquire")
        else:
            self.restore_to_gpu()

    def on_colocate_release(self, target: str) -> None:
        del target
        if self.world_size > 1 and self.is_master:
            self.submit_command("colocate_release")
        else:
            self.offload_to_cpu()

    def health_info(self) -> dict[str, Any]:
        return {"train_steps": getattr(self.critic, "train_steps", 0)}


__all__ = ["CriticSpmdEngine"]
