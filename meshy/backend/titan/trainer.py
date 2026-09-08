"""Minimal RL policy trainer built on torchtitan's ForgeEngine.

Design summary
--------------
* **One outer step = one ``train_step`` call.** No PPO inner epochs. A
  :class:`~meshy.backend.titan.plan.Plan` (built by :meth:`plan_batch` on
  every rank from the same global sample list) decides which samples this
  rank trains on and how they are grouped: one ``MiniPlan`` per
  ``optimizer.step()``, one ``MicroPlan`` per forward/backward. Micro-batches
  are sized by a token budget (``max_tokens_per_micro``) and run at their own
  sequence length instead of ``seq_len`` — see :mod:`meshy.backend.titan.plan`.
* **Two forward layouts** (:mod:`meshy.backend.titan.batch`): ``padded``
  keeps the causal-SDPA + ring-attention CP path; ``packed`` concatenates
  samples with ``cu_seqlens`` for variable-length attention (no CP).
* **Global normalisation.** Every micro-batch backpropagates
  ``local_sum / global_denominator`` where the denominator (number of
  sequences or of loss tokens in the mini-batch, over *all* DP ranks) comes
  from the plan. FSDP's gradient SUM across ``dp * cp`` then yields the exact
  global mean, whatever the per-rank sample / micro counts are.
* **CP** is hidden behind :class:`CpSharder`. Per-token tensors are opaque
  ``[rows, S_local]`` blobs with a ``doc_ids`` companion; per-sequence
  reductions are ``index_add_`` segment sums over ``doc_ids``.
* **Asymmetric PPO clip** with ``ppo_clip_eps_low`` and ``ppo_clip_eps_high``.
* **Behavior policy** for the importance ratio is configurable via
  ``old_logprobs_source``: ``"rollout"`` takes SGLang's per-token logprobs
  (already packed into the batch); ``"train"`` does an extra no-grad
  forward through the current training model over the same micro plan.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from torch.utils.checkpoint import checkpoint

from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.forge import ForgeEngine

from meshy.utils.timer import TimerStats

from .batch import Batch, build_micro_batch
from .compat import apply_forge_engine_compat
from .cp import CpSharder
from .plan import (
    Layout,
    MiniPlan,
    Plan,
    PlannerConfig,
    build_plan,
    plan_stats,
    resolve_align,
)


def _attn_backend_of(model_config: Any) -> str:
    """Name the inner-attention backend of a torchtitan decoder config."""
    try:
        inner = model_config.layers[0].attention.inner_attention
    except (AttributeError, IndexError, KeyError, TypeError):
        return "unknown"
    from torchtitan.models.common.attention import (
        FlexAttention,
        ScaledDotProductAttention,
        VarlenAttention,
    )

    if isinstance(inner, VarlenAttention.Config):
        return "varlen"
    if isinstance(inner, FlexAttention.Config):
        return "flex"
    if isinstance(inner, ScaledDotProductAttention.Config):
        return "sdpa"
    return type(inner).__name__


class TitanTrainer(ForgeEngine):
    """RL policy trainer with FSDP / TP / CP support (minimal cut)."""

    step: int
    seq_len: int
    micro_batch_size: int
    mini_batch_size: int
    batch_layout: Layout
    max_tokens_per_micro: int | None
    seq_align: int
    planner_config: PlannerConfig
    ppo_clip_eps_low: float
    ppo_clip_eps_high: float
    old_logprobs_source: Literal["rollout", "train"]
    calculate_per_token_loss: bool
    use_tis: bool
    tis_ratio_min: float
    tis_ratio_max: float
    log_entropy: bool
    entropy_chunk_size: int
    timer_enabled: bool
    sharder: CpSharder

    # 沿序列维度分块计算 log-prob 时的步长。控制 fp32 中间张量
    # ``[B, _LOGPROB_CHUNK, V]`` 的峰值大小：越小越省显存，但 kernel
    # launch 次数越多。1024 在 V≈150K 的 Qwen3 上是个不错的折中。
    _LOGPROB_CHUNK: int = 1024

    def __init__(
        self,
        job_config: ForgeEngine.Config,
        *,
        micro_batch_size: int = 1,
        mini_batch_size: int = 1,
        batch_layout: Layout = "padded",
        max_tokens_per_micro: int | None = None,
        seq_bucket: int = 2048,
        ppo_clip_eps_low: float = 0.2,
        ppo_clip_eps_high: float = 0.2,
        old_logprobs_source: Literal["rollout", "train"] = "rollout",
        calculate_per_token_loss: bool = False,
        use_tis: bool = False,
        tis_ratio_min: float = 0.5,
        tis_ratio_max: float = 5.0,
        logprob_chunk_size: int = 1024,
        log_entropy: bool = True,
        entropy_chunk_size: int = 512,
        timer_enabled: bool = True,
    ) -> None:
        # torchtitan's forge engine still expects the pre-refactor loss
        # plumbing (``ModelSpec.loss``) and reads an unbound
        # ``parallelism_config``; patch both in before the engine runs. See
        # :mod:`meshy.backend.titan.compat`.
        apply_forge_engine_compat(job_config)
        super().__init__(job_config)

        self.step = 0
        self.seq_len = self.config.training.seq_len
        self.micro_batch_size = max(1, micro_batch_size)
        self.mini_batch_size = max(self.micro_batch_size, mini_batch_size)
        self.ppo_clip_eps_low = ppo_clip_eps_low
        self.ppo_clip_eps_high = ppo_clip_eps_high
        # Validated upstream by ``meshy.config.TrainerParamsConfig``.
        self.old_logprobs_source = old_logprobs_source
        self.calculate_per_token_loss = bool(calculate_per_token_loss)
        self.use_tis = bool(use_tis)
        self.tis_ratio_min = float(tis_ratio_min)
        self.tis_ratio_max = float(tis_ratio_max)
        if self.use_tis and not (0.0 < self.tis_ratio_min <= self.tis_ratio_max):
            raise ValueError("TIS ratio bounds must satisfy 0 < min <= max")
        if logprob_chunk_size <= 0:
            raise ValueError("logprob_chunk_size must be positive")
        self._LOGPROB_CHUNK = int(logprob_chunk_size)
        if entropy_chunk_size <= 0:
            raise ValueError("entropy_chunk_size must be positive")
        self.log_entropy = bool(log_entropy)
        self.entropy_chunk_size = int(entropy_chunk_size)
        # Mirrors ``GRPOPipeline.timer_enabled`` — wired in by the
        # pipeline from the top-level ``config["timer"]["enabled"]``
        # flag. When False, the per-step ``TimerStats`` built inside
        # :meth:`train_step` becomes a no-op and emits no
        # ``time/train/*`` metrics, skipping the ``cuda.synchronize``
        # calls that would otherwise serialise every stage.
        self.timer_enabled = bool(timer_enabled)
        self.sharder = CpSharder(
            self.parallel_dims,
            load_balancer=self.config.parallelism.context_parallel_load_balancer,
        )
        # torchtitan's parallelize already asserts seq_len % (tp * cp * 2);
        # every dynamic micro-batch length must satisfy the same divisor.
        divisor = self.parallel_dims.seq_len_divisor
        self.batch_layout = batch_layout
        self.max_tokens_per_micro = (
            int(max_tokens_per_micro) if max_tokens_per_micro else None
        )
        self.seq_align = resolve_align(int(seq_bucket), divisor, self.seq_len)
        attn_backend = _attn_backend_of(self.config.model_spec.model)
        if self.batch_layout == "packed":
            if attn_backend != "varlen":
                raise ValueError(
                    "batch_layout='packed' requires attn_backend='varlen' "
                    f"(model uses {attn_backend}); SDPA has no document mask"
                )
            if self.sharder.enabled:
                raise NotImplementedError(
                    "batch_layout='packed' does not support context parallelism "
                    "yet (torchtitan has no varlen CP); use 'padded' with cp>1"
                )
        elif attn_backend != "sdpa":
            raise ValueError(
                "batch_layout='padded' expects attn_backend='sdpa' "
                f"(model uses {attn_backend})"
            )
        self.planner_config = PlannerConfig(
            layout=self.batch_layout,
            mini_batch_size=self.mini_batch_size,
            seq_len=self.seq_len,
            align=self.seq_align,
            max_tokens_per_micro=self.max_tokens_per_micro,
            micro_batch_size=self.micro_batch_size,
        )
        if self.max_tokens_per_micro is not None and self.micro_batch_size > 1:
            logger.warning(
                "max_tokens_per_micro={} is set; micro_batch_size={} is ignored "
                "(micro-batches are sized by tokens).",
                self.max_tokens_per_micro, self.micro_batch_size,
            )
        if self.config.compile.enable:
            # Micro-batch shapes now vary per step. Dynamo turns the block
            # graphs dynamic after the second distinct shape, but the default
            # cache of 8 entries is too small for the bucketed shape set.
            import torch._dynamo

            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 64
            )

        loaded = self.checkpointer.load(step=self.config.checkpoint.load_step)
        if self.config.checkpoint.initial_load_path and not loaded:
            # load() returning False is a *legitimate* outcome for torchtitan's
            # native pretraining scenario ("nothing to resume, start from
            # random init"), so it never raises on its own -- e.g. it returns
            # False straight away when checkpoint.enable is off, silently
            # skipping the initial_load_path branch. For this trainer a
            # configured base model is mandatory: proceeding would train (and
            # publish to inference!) a randomly initialized model whose dumped
            # checkpoints look perfectly valid, with garbled generations as
            # the only symptom. Fail at startup instead.
            raise RuntimeError(
                "initial_load_path is configured "
                f"({self.config.checkpoint.initial_load_path!r}) but "
                "CheckpointManager.load() loaded nothing; the model would run "
                "with random initialization. Check that checkpoint.enable is "
                "True and the path is a valid checkpoint."
            )

        logger.info(
            "TitanTrainer initialized: params={:,}, seq_len={}, "
            "dp_degree={}, tp={}, cp={}, layout={}, attn={}, micro={}, "
            "max_tokens_per_micro={}, seq_align={}, mini={}, "
            "clip=[{:.3f}, {:.3f}], old_logprobs_source={}",
            self.model_param_count, self.seq_len, self.dp_degree,
            self.parallel_dims.tp, self.parallel_dims.cp,
            self.batch_layout, attn_backend, self.micro_batch_size,
            self.max_tokens_per_micro, self.seq_align, self.mini_batch_size,
            1 - self.ppo_clip_eps_low, 1 + self.ppo_clip_eps_high,
            self.old_logprobs_source,
        )

        # torch.compile interactions with this trainer's GPU lifecycle
        # ----------------------------------------------------------------
        # When ``build_forge_config(compile_model=True)`` is set, torchtitan
        # has already wrapped each ``TransformerBlock`` in-place with
        # ``torch.compile(backend=..., fullgraph=True)`` via
        # ``apply_compile_sparse`` (called from ``parallelize_qwen3`` before
        # FSDP / DDP wrapping). ``self.model_parts`` therefore already
        # contains compiled modules; do NOT wrap them again here.
        #
        # The RL training loop offloads weights to CPU and restores them to
        # GPU around every ``train_step`` (see :meth:`offload_to_cpu` /
        # :meth:`restore_to_gpu`). This is safe with the default
        # ``backend="inductor"`` configuration:
        #   * Dynamo / Inductor guards key on parameter ``shape / dtype /
        #     device / stride`` (not pointer identity). After ``restore_to_gpu``
        #     the device is back to CUDA and shapes are unchanged, so the
        #     compiled cache hits and no recompile fires.
        #   * Inductor without an explicit ``mode="reduce-overhead"`` does
        #     NOT enable CUDA Graphs, so there is no static-input pointer
        #     capture to invalidate.
        # If a future change opts into CUDA Graphs (``mode="reduce-overhead"``
        # or ``torch._inductor.config.triton.cudagraphs = True``), the
        # per-step ``.to(device)`` will rewrite parameter storage pointers
        # and force a graph re-capture on every first-forward post-restore —
        # at that point either disable cudagraphs or rework offload to use
        # in-place ``copy_`` so parameter pointers stay stable.
        #
        # Note: ``_forward_logprobs`` deliberately keeps its chunked
        # ``F.cross_entropy`` outside the compile boundary. The matmul-heavy
        # work inside ``model(input_ids, positions=positions)`` is already
        # fused per block, and pulling the chunk loop into compile would risk
        # graph breaks under loss_parallel without a meaningful speedup.

    # ------------------------------------------------------------------
    # Memory management  (GPU ↔ CPU)
    # ------------------------------------------------------------------

    def _move_optimizer_states(self, device: str | torch.device) -> None:
        """Move every optimizer state tensor (exp_avg / exp_avg_sq) to *device*.

        Optimizer states are the largest single residency of the training
        process after the first ``optimizer.step()`` -- for AdamW, two fp32
        tensors per parameter, i.e. ~2x the sharded parameter bytes per rank.
        They are stored in ``optimizer.state`` keyed by parameter object, so
        ``model.to(...)`` does NOT move them: without this, they silently stay
        on the GPU across every colocate hand-off, permanently stealing that
        memory from inference *and* from the next step's activations (observed
        as a step-2 OOM at 30B scale: step 1 runs before the states exist,
        step 2 starts ~30 GB/rank deeper).

        ``step`` counters are skipped: they are scalar tensors whose device is
        an implementation detail of the AdamW variant in use (plain vs
        fused/capturable), and moving them can break those kernels' device
        assumptions. Under FSDP2 the states are DTensors; ``.to`` transfers
        the local shard and round-trips cleanly.
        """
        target = torch.device(device)
        for optimizer in self.optimizers:
            for state in optimizer.state.values():
                for key, value in state.items():
                    if key == "step" or not torch.is_tensor(value):
                        continue
                    if value.device.type != target.type:
                        state[key] = value.to(target)

    def _refresh_checkpointer_cache(self) -> None:
        """Re-point the checkpointer's cached state dict at current storages.

        torchtitan's ``ModelWrapper`` (held in ``checkpointer.states``)
        snapshots state-dict tensor *references* at build/load time.
        ``Module.to()`` rebinds every ``param.data`` to a fresh tensor on the
        target device, so after a device move that cache pins the old device's
        tensors alive -- one full sharded-model copy leaked on the GPU per
        offload, forever. Refreshing after every move drops the stale
        references and keeps any later DCP save pointed at live storage.
        """
        for state in self.checkpointer.states.values():
            if hasattr(state, "cache_state_dict") and hasattr(state, "_get_state_dict"):
                state.cache_state_dict = state._get_state_dict()

    def offload_to_cpu(self) -> None:
        """Move the full training residency (params + optimizer states) to CPU.

        Call this **before** the inference engine (e.g. SGLang) reclaims the
        GPU. Both the model parameters and the optimizer states move; leaving
        the optimizer behind would keep ~2x the sharded parameter bytes per
        rank resident for the rest of the run.
        """
        for model_part in self.model_parts:
            model_part.to("cpu")
        self._move_optimizer_states("cpu")
        self._refresh_checkpointer_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("TitanTrainer: model + optimizer states offloaded to CPU.")

    def offload_optimizer_to_cpu(self) -> None:
        """Park only the optimizer states on CPU, leaving the model on GPU.

        The optimizer states are the largest remaining residency after a
        forward/optimizer step, so this helper can release them independently
        when a caller needs to keep model parameters resident.
        """
        self._move_optimizer_states("cpu")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("TitanTrainer: optimizer states offloaded to CPU.")

    def restore_to_gpu(self) -> None:
        """Move model parameters and optimizer states back to the training GPU.

        Call this **after** the inference engine has released GPU memory
        and before :meth:`train_step`.
        """
        for model_part in self.model_parts:
            model_part.to(self.device)
        self._move_optimizer_states(self.device)
        self._refresh_checkpointer_cache()
        logger.info(f"TitanTrainer: model + optimizer states restored to {self.device}.")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_logprobs(
        logits_chunk: torch.Tensor,  # [B, T, V] (DTensor under TP)
        labels_chunk: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        B, T = labels_chunk.shape
        # cross_entropy expects [N, C] + [N]; reshape preserves the
        # vocab dim as the (sharded) last dim so loss_parallel can
        # dispatch the TP-aware kernel.
        nll = F.cross_entropy(
            logits_chunk.float().reshape(B * T, -1),
            labels_chunk.reshape(B * T),
            reduction="none",
        ).view(B, T)
        return -nll

    @torch.no_grad()
    def _chunk_entropy(self, logits_chunk: torch.Tensor) -> torch.Tensor:
        """Per-position softmax entropy of a ``[rows, T, V]`` logits slice (fp32).

        Processed in ``entropy_chunk_size`` token slices so the fp32 transient
        stays at ``[rows, entropy_chunk_size, V]``. Not TP-aware: a vocab-
        sharded DTensor is gathered first (``full_tensor``) -- TP recipes pay
        one all-gather per slice, which is why this path is optional.
        """
        if hasattr(logits_chunk, "full_tensor"):
            logits_chunk = logits_chunk.full_tensor()
        out: list[torch.Tensor] = []
        step = self.entropy_chunk_size
        for s in range(0, logits_chunk.size(1), step):
            z = logits_chunk[:, s:s + step, :].float()
            lse = torch.logsumexp(z, dim=-1)
            # H = logsumexp(z) - sum(softmax(z) * z)
            out.append(lse - (torch.softmax(z, dim=-1) * z).sum(dim=-1))
            del z, lse
        return torch.cat(out, dim=1)

    def _forward_logprobs(
        self,
        input_ids: torch.Tensor,   # [rows, S_local]
        labels: torch.Tensor,       # [rows, S_local], next-token, shifted
        positions: torch.Tensor,    # [rows, S_local], RoPE positions
        attention_masks: Any = None,  # VarlenMetadata for ``packed``; None otherwise
        *,
        with_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run model forward and gather per-position log-probs of labels.

        With ``with_entropy=True`` the per-position policy entropy (detached,
        fp32, same ``[rows, S_local]`` layout) is returned alongside.

        Inputs are *already* laid out (and CP-sharded) by
        ``build_micro_batch``. We compute ``log p(y) = -cross_entropy(logits,
        y, reduction='none')`` in fp32 along the local sequence dim, chunked
        to bound the peak fp32 buffer to ``[rows, _LOGPROB_CHUNK, V_local]``.

        Each chunk runs under non-reentrant activation checkpointing while
        grad is enabled. Without it, chunking bounds nothing on the training
        path: ``cross_entropy`` saves its fp32 input for backward, so every
        chunk's ``[rows, T, V]`` fp32 upcast stays resident until the backward
        pass consumes it and the peak is the *whole* fp32 logits tensor
        (32k x 130k vocab -> 17 GiB per micro-batch at 128k context under
        CP=4). Recomputing the upcast + cross_entropy in backward keeps only
        the bf16 ``logits`` alive between the two passes. Recomputation runs
        inside the caller's ``train_context()`` (``backward()`` is invoked
        there too), so the ``loss_parallel`` dispatch below stays in effect.

        Why ``cross_entropy`` instead of ``gather - logsumexp``:
        * Under ``loss_parallel()`` (entered by ``self.train_context``
          whenever ``tp_enabled and not disable_loss_parallel``), the
          LM head emits a DTensor sharded on the vocab dim. The naive
          ``gather`` / ``logsumexp`` along that dim are NOT TP-aware and
          silently return wrong values. ``F.cross_entropy`` is patched
          by ``loss_parallel`` to do the vocab-sharded all-reduces
          (MAX for the log-sum-exp shift, SUM for the target-logit
          gather), so this kernel is correct on both plain Tensors and
          on Shard(-1) DTensors.
        * Math is identical: ``-CE(z, y) = z_y - logsumexp(z) = log p(y)``.
        * Autograd does not save the fp32 softmax buffer (it recomputes
          ``softmax(z) - onehot(y)`` on backward), so peak memory is
          lower than the explicit two-pass version.
        """
        model = self.model_parts[0]
        logits = model(
            input_ids, positions=positions, attention_masks=attention_masks
        )  # [rows, S_local, V] or DTensor

        S_local = logits.size(1)
        chunk = self._LOGPROB_CHUNK if S_local > self._LOGPROB_CHUNK else S_local

        recompute = torch.is_grad_enabled() and logits.requires_grad

        out_chunks: list[torch.Tensor] = []
        ent_chunks: list[torch.Tensor] = []
        for s in range(0, S_local, chunk):
            e = min(s + chunk, S_local)
            sl = logits[:, s:e, :]                           # [rows, T, V] view
            tg = labels[:, s:e]                              # [rows, T]
            if recompute:
                lp = checkpoint(
                    self._chunk_logprobs, sl, tg, use_reentrant=False,
                )
            else:
                lp = self._chunk_logprobs(sl, tg)
            out_chunks.append(lp)
            if with_entropy:
                ent_chunks.append(self._chunk_entropy(sl.detach()))
            del sl, lp

        token_lps = torch.cat(out_chunks, dim=1)
        del out_chunks, logits
        if with_entropy:
            return token_lps, torch.cat(ent_chunks, dim=1)
        return token_lps

    def _forward_batch(
        self, mb: Batch, *, with_entropy: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._forward_logprobs(
            mb.input_ids, mb.labels, mb.positions, mb.attention_masks,
            with_entropy=with_entropy,
        )

    @torch.no_grad()
    def _compute_old_logprobs_train(
        self, samples: Sequence[Any], mini: MiniPlan
    ) -> list[torch.Tensor]:
        """Run the training model over ``mini``'s micro plan (no grad) → ``old_lp``.

        Uses exactly the micro-batches the PG pass will use so
        ``new_lp - old_lp`` lines up token for token. The same
        ``train_context`` is entered to keep autocast/loss_parallel/CP
        dispatch identical to the PG forward — important for BF16 numerical
        agreement (``ratio`` should be ~1 at step 0).
        """
        out: list[torch.Tensor] = []
        with self.train_context():
            for micro in mini.micros:
                mb = self._build_micro(samples, micro, need_rollout_lp=False)
                out.append(self._forward_batch(mb))
                del mb
        return out

    def _build_micro(
        self, samples: Sequence[Any], micro, *, need_rollout_lp: bool | None = None
    ) -> Batch:
        if need_rollout_lp is None:
            # Rollout log-probs are the behaviour policy for ``"rollout"`` and
            # feed the train-vs-rollout mismatch metrics for ``"train"``.
            need_rollout_lp = True
        return build_micro_batch(
            samples,
            micro,
            layout=self.batch_layout,
            device=self.device,
            sharder=self.sharder,
            need_rollout_lp=need_rollout_lp,
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_lengths(samples: Sequence[Any], seq_len: int) -> tuple[list[int], list[int]]:
        """Per-sample ``(token count, loss-token count after shift + truncation)``."""
        lengths: list[int] = []
        loss_tokens: list[int] = []
        for td in samples:
            n = int(len(td["tokens"]))
            L = min(n, seq_len)
            lengths.append(n)
            mask = td["mask_assistant"]
            loss_tokens.append(int((mask[1:L] > 0).sum().item()) if L > 1 else 0)
        return lengths, loss_tokens

    def plan_batch(self, samples: Sequence[Any], dp_size: int) -> Plan:
        """Build the dynamic-batching plan for the *global* sample list.

        Deterministic in ``samples`` and the trainer config, so every rank
        calls it on the broadcast batch and gets the same plan.
        """
        lengths, loss_tokens = self._sample_lengths(samples, self.seq_len)
        return build_plan(lengths, loss_tokens, dp_size, self.planner_config)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _ppo_clip_loss(
        self,
        new_lp: torch.Tensor,   # [rows, S_local]
        old_lp: torch.Tensor,   # [rows, S_local]
        mb: Batch,
        mini: MiniPlan,
        entropy: torch.Tensor | None = None,  # [rows, S_local] or None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Asymmetric-clipped PPO policy-gradient loss for one micro-batch.

        Returns the loss to backpropagate — this rank's **partial** sum
        already divided by the mini-batch's *global* denominator — plus
        detached sums for metrics (reduced once per mini-batch).

        Diagnostics reported alongside the loss (all masked to loss tokens,
        token sums so the mini-batch reduce can turn them into means):

        * ``ppo_kl_sum``: ``old_lp - new_lp`` (Miles ``train/ppo_kl``).
        * ``train_rollout_kl_sum``: Schulman k3 estimate of
          ``KL(rollout || train)`` at the sampled tokens between SGLang's
          log-probs and the training model (``old_lp`` when the behaviour
          policy is recomputed by the trainer, else ``new_lp``). Needs
          ``mb.rollout_logprobs``.
        * ``entropy_sum``: policy entropy when ``entropy`` is given.
        * ``ess_sum``: per-sequence effective-sample-size ratio
          ``(sum w)^2 / (n sum w^2)`` of the importance weights, summed over
          the sequences of this rank (already CP-global, hence divided by the
          CP degree so the later SUM over ``dp x cp`` counts it once).

        Aggregation (GRPO-style sequence mean by default, optional token mean):
        * Per-token importance ratio ``exp(new_lp - old_lp)`` (not
          ``exp(sum)``) so gradient magnitude is length-independent.
        * Sequence mean: per-token PG loss is averaged within each sequence
          (segment sum over ``doc_ids`` / global token count of that
          sequence) and the per-sequence means are summed and divided by
          ``mini.n_docs_global``. ``calculate_per_token_loss`` switches to
          ``sum / mini.n_tokens_global``.
        * Under CP the per-sequence token counts are all-reduced (no grad);
          the numerator stays local and FSDP's SUM over ``dp_shard * cp``
          assembles the global gradient. Under DP the same SUM assembles
          the mean over all ranks' samples.

        ``mb.advantages`` may be per-sequence (GRPO, one value broadcast
        across the sequence) or already per-token ``[rows, S_local]``.
        """
        mask = mb.mask
        log_ratio = new_lp - old_lp
        ratio = log_ratio.exp()
        adv = mb.advantages
        if adv.dim() == 1:
            # Slot ``n_docs`` (padding) gets advantage 0.
            adv_tok = torch.cat([adv, adv.new_zeros(1)])[mb.doc_ids]
        else:
            adv_tok = adv

        pg_unclipped = ratio * adv_tok
        pg_clipped = torch.clamp(
            ratio, 1 - self.ppo_clip_eps_low, 1 + self.ppo_clip_eps_high,
        ) * adv_tok
        pg_per_token = -torch.min(pg_unclipped, pg_clipped)
        if self.use_tis:
            tis_mask = (ratio >= self.tis_ratio_min) & (ratio <= self.tis_ratio_max)
            # ``where`` avoids ``0 * inf -> nan`` for pathological log-ratios.
            pg_per_token = torch.where(tis_mask, pg_per_token, torch.zeros_like(pg_per_token))
        else:
            tis_mask = torch.ones_like(ratio, dtype=torch.bool)
        pg_per_token = pg_per_token * mask

        flat_ids = mb.doc_ids.reshape(-1)
        seg_sum = pg_per_token.new_zeros(mb.n_slots).index_add_(
            0, flat_ids, pg_per_token.reshape(-1)
        )
        seg_cnt = mask.new_zeros(mb.n_slots).index_add_(0, flat_ids, mask.reshape(-1))
        seg_cnt_global = self.sharder.all_reduce_sum(seg_cnt)
        n = mb.n_docs
        if self.calculate_per_token_loss:
            loss = seg_sum[:n].sum() / max(1, mini.n_tokens_global)
        else:
            loss = (seg_sum[:n] / seg_cnt_global[:n].clamp(min=1)).sum() / max(
                1, mini.n_docs_global
            )

        with torch.no_grad():
            eps = max(self.ppo_clip_eps_low, self.ppo_clip_eps_high)
            ratio_d = ratio.detach()
            log_ratio_d = log_ratio.detach()
            sums = {
                "loss": loss.detach(),
                "ratio_sum": (ratio_d * mask).sum(),
                "clip_sum": (((ratio_d - 1).abs() > eps).float() * mask).sum(),
                "tis_masked_sum": ((~tis_mask).float() * mask).sum(),
                "token_count": mask.sum(),
                "ppo_kl_sum": (-log_ratio_d * mask).sum(),
                "log_ratio_abs_sum": (log_ratio_d.abs() * mask).sum(),
            }
            # Effective sample size of the importance weights, per sequence.
            cp_degree = max(1, int(getattr(self.parallel_dims, "cp", 1)))
            seg_w = ratio_d.new_zeros(mb.n_slots).index_add_(0, flat_ids, (ratio_d * mask).reshape(-1))
            seg_w2 = ratio_d.new_zeros(mb.n_slots).index_add_(
                0, flat_ids, (ratio_d * ratio_d * mask).reshape(-1)
            )
            seg_w = self.sharder.all_reduce_sum(seg_w)[:n]
            seg_w2 = self.sharder.all_reduce_sum(seg_w2)[:n]
            cnt = seg_cnt_global[:n]
            ess = (seg_w * seg_w) / (cnt.clamp(min=1) * seg_w2.clamp(min=1e-8))
            sums["ess_sum"] = ess[cnt > 0].sum() / cp_degree
            sums["doc_count"] = (cnt > 0).float().sum() / cp_degree
            if mb.rollout_logprobs is not None:
                train_lp = old_lp if self.old_logprobs_source == "train" else new_lp.detach()
                # k3 estimator of KL(rollout || train), clamped like the baseline.
                lr_rt = (mb.rollout_logprobs - train_lp).clamp(-10.0, 10.0)
                k3 = (lr_rt.exp() - lr_rt - 1.0)
                sums["train_rollout_kl_sum"] = (torch.nan_to_num(k3) * mask).sum()
                sums["train_rollout_logdiff_abs_sum"] = (
                    torch.nan_to_num((mb.rollout_logprobs - train_lp).abs()) * mask
                ).sum()
            if entropy is not None:
                sums["entropy_sum"] = (entropy.float() * mask).sum()
        return loss, sums

    # ------------------------------------------------------------------
    # Training step  (public API)
    # ------------------------------------------------------------------

    def train_step(
        self,
        samples: list[Any],
        *,
        plan: Sequence[MiniPlan] | None = None,
        step_schedule: bool = True,
    ) -> dict[str, float]:
        """Perform one outer training step over this rank's ``samples``.

        ``plan`` is this rank's slice of the global :class:`Plan` (see
        :meth:`plan_batch` and ``split_batch_to_local``) and ``samples`` the
        matching rank-local list (``Plan.local_samples``). When ``plan`` is
        ``None`` the rank plans ``samples`` on its own as a single-rank job,
        which is what ``world_size == 1`` and the unit tests want.

        Layout per mini-batch (one ``optimizer.step()`` each):
        1. Snapshot the behavior-policy log-probs for the mini-batch's
           micro plan — rollout log-probs come with the samples, ``"train"``
           runs a no-grad forward over the same micro-batches *before* the
           optimizer moves.
        2. For every micro-batch: build it, forward, PPO loss normalised by
           the mini-batch's global denominators, backward.
        3. Clip + step.

        ``step_schedule`` controls whether this call advances the LR
        scheduler and the outer ``self.step`` counter. It is ``True`` for a
        normal (whole-batch) step. Under the ``stream_minibatch`` worker
        schedule the trainer is fed one ``mini_batch_size`` chunk at a time;
        all but the batch-closing chunk pass ``step_schedule=False`` so the
        optimizer still steps per chunk but the LR schedule / step counter
        only advance once per ``batch_size`` (i.e. per version), keeping the
        LR curve and ``steps`` horizon identical to the non-streamed mode.

        Internal stage times are accumulated into ``time/train/*`` and
        appended to the returned metrics dict so the caller can ship
        them to its dashboard alongside the loss-side metrics.
        """
        timer = TimerStats(enabled=self.timer_enabled)

        if plan is None:
            with timer.timer("train/plan"):
                full = self.plan_batch(samples, 1)
                samples = full.local_samples(samples, 0)
                plan = full.per_rank[0]

        all_metrics: list[dict[str, float]] = []
        grad_norms: list[torch.Tensor] = []
        for mini in plan:
            mini_metrics, grad_norm = self._run_mini_batch(samples, mini, timer)
            all_metrics.append(mini_metrics)
            grad_norms.append(grad_norm)

        with timer.timer("train/lr_step"):
            if step_schedule:
                self.lr_schedulers.step()
        if step_schedule:
            self.step += 1

        result = self._aggregate_metrics(all_metrics, grad_norms, timer)
        for key, value in plan_stats(plan, self.batch_layout).items():
            result[f"train/{key}"] = float(value)
        result["num_mini_batches"] = float(len(plan))
        return result

    def _run_mini_batch(
        self,
        samples: Sequence[Any],
        mini: MiniPlan,
        timer: TimerStats,
    ) -> tuple[dict[str, float], "torch.Tensor"]:
        """Accumulate gradients over ``mini``'s micro-batches, then one optimizer step.

        Every micro-batch's loss is already scaled by the mini-batch's global
        denominator, so the micro losses are simply summed (no ``1/n_micro``).
        Returns the mini-batch's globally reduced metrics and gradient norm.
        """
        with timer.timer("train/old_logprobs", sync=True):
            if self.old_logprobs_source == "train":
                old_lps: list[torch.Tensor | None] = list(
                    self._compute_old_logprobs_train(samples, mini)
                )
            else:
                old_lps = [None] * len(mini.micros)

        sums: dict[str, torch.Tensor] | None = None
        self.optimizers.zero_grad()
        for micro, old_lp in zip(mini.micros, old_lps):
            with timer.timer("train/pack_batch", sync=True):
                mb = self._build_micro(samples, micro)
            if old_lp is None:
                assert mb.rollout_logprobs is not None, (
                    "old_logprobs_source='rollout' but the batch carries no "
                    "rollout log-probs"
                )
                old_lp = mb.rollout_logprobs
            with self.train_context():
                with timer.timer("train/forward", sync=True):
                    entropy = None
                    if self.log_entropy:
                        new_lp, entropy = self._forward_batch(mb, with_entropy=True)
                    else:
                        new_lp = self._forward_batch(mb)
                    loss, mb_sums = self._ppo_clip_loss(
                        new_lp, old_lp, mb, mini, entropy=entropy
                    )
                    del new_lp, entropy
                with timer.timer("train/backward", sync=True):
                    loss.backward()
            if sums is None:
                sums = mb_sums
            else:
                sums = {k: sums[k] + v for k, v in mb_sums.items()}
            del mb, old_lp, loss, mb_sums

        # ppo 内 microbatch 跑完后把 cached 但未占用的 block 真正归还给
        # CUDA driver。后续 sglang 等同卡进程要拿显存时这一步是关键。
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
        """Turn per-rank sums into global (over ``batch x cp``) mini-batch metrics."""
        keys = list(sums)
        stacked = torch.stack([sums[k].float() for k in keys])
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")
        if loss_mesh is not None:
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=loss_mesh.get_group())
        values = dict(zip(keys, stacked.tolist()))
        tokens = max(values["token_count"], 1.0)
        out = {
            "pg_loss": values["loss"],
            "ratio_mean": values["ratio_sum"] / tokens,
            "clip_frac": values["clip_sum"] / tokens,
            "tis_masked_frac": values["tis_masked_sum"] / tokens,
            "ppo_kl": values["ppo_kl_sum"] / tokens,
            "log_ratio_abs_mean": values["log_ratio_abs_sum"] / tokens,
            "ess_ratio": values["ess_sum"] / max(values.get("doc_count", 0.0), 1.0),
        }
        if "train_rollout_kl_sum" in values:
            out["train_rollout_kl"] = values["train_rollout_kl_sum"] / tokens
            out["train_rollout_logdiff_abs"] = values["train_rollout_logdiff_abs_sum"] / tokens
        if "entropy_sum" in values:
            out["entropy"] = values["entropy_sum"] / tokens
        return out

    def _aggregate_metrics(
        self,
        all_metrics: list[dict[str, float]],
        grad_norms: "Sequence[torch.Tensor | None]",
        timer: TimerStats,
    ) -> dict[str, float]:
        """Mean the per-mini-batch metrics and attach grad_norm / step / timings.

        ``grad_norm`` is the mean pre-clip gradient norm over the step's
        optimizer updates (one per mini-batch); ``grad_norm_max`` / ``_last``
        keep the extremes visible. Note the norm is measured per *mini-batch*
        (``mini_batch_size x dp`` samples), so its scale depends on that size,
        not on the rollout batch.
        """
        result: dict[str, float] = {}
        for key in all_metrics[0]:
            result[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        norms = [float(g.item()) for g in grad_norms if g is not None]
        result["grad_norm"] = sum(norms) / len(norms) if norms else 0.0
        result["grad_norm_max"] = max(norms) if norms else 0.0
        result["grad_norm_last"] = norms[-1] if norms else 0.0
        result["step"] = self.step
        # Inline internal stage timings (``time/train/...``) so the
        # caller can route them to its timing dashboard.
        result.update(timer.pop_metrics(prefix="time/"))
        return result

    # ------------------------------------------------------------------
    # Checkpoint & weight sync helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, last_step: bool = False) -> None:
        """Persist a DCP checkpoint at the current step."""
        self.checkpointer.save(self.step, last_step=last_step)

    def save_hf_checkpoint(self, output_dir: str) -> None:
        """Export current weights to a HuggingFace-format directory.

        All ranks must call this together — the gather is a distributed
        all-gather. Only rank 0 writes files. A barrier ensures callers on
        every rank see the files on return.
        """
        self.save_hf_state_dict(self.gather_hf_state_dict(cpu_offload=True), output_dir)

    def save_hf_state_dict(
        self, state_dict: dict[str, torch.Tensor], output_dir: str
    ) -> None:
        """Write an already-gathered HF state dict out as a checkpoint.

        Split out of :meth:`save_hf_checkpoint` so callers that already hold a
        gathered state dict can persist it without paying for a second
        all-gather. All ranks must call together
        (the closing barrier is collective); only rank 0 touches the disk, and
        the dict is left intact for the caller to go on using.
        """
        import os

        from safetensors.torch import save_file
        from transformers import AutoConfig, AutoTokenizer

        if dist.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)
            weight_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, weight_path)
            logger.info(f"save_hf_checkpoint: weights → {weight_path}")
            hf_src = self.config.hf_assets_path
            if hf_src:
                AutoConfig.from_pretrained(hf_src).save_pretrained(output_dir)
                AutoTokenizer.from_pretrained(hf_src).save_pretrained(output_dir)
            logger.info(f"save_hf_checkpoint: done → {output_dir}")

        dist.barrier()

    def gather_hf_state_dict(self, cpu_offload: bool = True) -> dict[str, torch.Tensor]:
        """Gather the full (un-sharded) state dict with HF parameter names.

        Must be called while the model is still on GPU. All ranks must
        call together.

        Args:
            cpu_offload: When True, gathered tensors are streamed to CPU
                RAM (lower GPU peak; needs a copy back for re-use) and --
                this is torch's rule, not ours -- **only rank 0 receives
                them**: ``full_state_dict and cpu_offload`` makes DCP pass
                ``ranks_only=(0,)``. Fine for the disk path, where only rank 0
                writes. When False, every rank gets the full dict on its own
                GPU.
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        logger.info(
            "gather_hf_state_dict: full gather (cpu_offload={}, lands on {}) ...",
            cpu_offload,
            "rank 0 only" if cpu_offload else "every rank",
        )
        state_dict = get_model_state_dict(
            self.model_parts[0],
            options=StateDictOptions(
                full_state_dict=True, cpu_offload=cpu_offload,
            ),
        )
        sd_adapter = self.config.model_spec.state_dict_adapter
        if sd_adapter is not None:
            adapter = sd_adapter(self.config.model_spec.model, None)
            state_dict = adapter.to_hf(state_dict)
        logger.info(
            "gather_hf_state_dict: done, {} params (rank {})",
            len(state_dict), dist.get_rank(),
        )
        return state_dict

    def get_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the (possibly FSDP-sharded) model state dict."""
        sd: dict[str, torch.Tensor] = {}
        for part in self.model_parts:
            sd.update(part.state_dict())
        return sd

    # ------------------------------------------------------------------
    # Stateful protocol  (consumed by CheckpointManager)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"]
