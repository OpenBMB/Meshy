"""Typed configuration contract for the Meshy runtime.

Meshy intentionally owns this schema. The old xrl role configs are not
re-exported: removed roles and removed CUDA-IPC options fail at recipe
construction time instead of surviving as misleading fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Literal


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 16384

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class GlobalCheckpointConfig:
    """How often the run persists its global state, and how much it keeps.

    A Meshy run has no driver, so there is no process that could write "the"
    checkpoint. Each role persists its own shard on a shared clock -- the
    global window index -- and the resume point is *derived* as the newest step
    at which every participant committed a shard (see
    :mod:`meshy.service.checkpoint`).

    One instance is shared by every checkpointing Service in a recipe. The
    topology rejects a recipe whose services disagree, because the resume point
    is an intersection over shards: services saving on different clocks would
    simply never intersect.
    """

    enabled: bool = False
    #: Global windows between checkpoints. Every participant uses the same
    #: ``global_step % interval == 0`` test, which is what aligns their shards.
    interval: int = 10
    #: Complete checkpoints retained per service. Must be >= 2: the newest one
    #: may be half-written by a peer that has not reached this step yet, so a
    #: single retained step can leave the intersection empty.
    keep_latest_k: int = 3

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError(f"checkpoint interval must be positive, got {self.interval!r}")
        if self.keep_latest_k < 2:
            raise ValueError(
                "checkpoint keep_latest_k must be >= 2; with 1 the only retained "
                f"step can be one a peer has not written yet (got {self.keep_latest_k!r})"
            )

    def due(self, global_step: int) -> bool:
        """Whether ``global_step`` is a checkpoint boundary."""
        return self.enabled and global_step > 0 and global_step % self.interval == 0

    @classmethod
    def coerce(cls, value: "GlobalCheckpointConfig | dict | None") -> "GlobalCheckpointConfig":
        """Accept an instance, a ``dataclasses.asdict`` dict, or ``None``.

        The rollout Service flattens its whole typed config with ``asdict``
        before handing it to the driver subprocess, which turns nested
        dataclasses into plain dicts.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**value)


@dataclass
class TrainerConfig:
    model_name: str = "qwen3"
    model_flavor: str = "1.7B"
    seq_len: int = 2048
    steps: int = 100
    dtype: str = "bfloat16"
    max_norm: float = 1.0
    local_batch_size: int = 4
    global_batch_size: int = -1
    lr: float = 1e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    warmup_steps: int = 0
    # LR schedule after warmup. torchtitan's pretraining default
    # (``decay_ratio=None``) starts a linear decay to ``min_lr_factor * lr``
    # right after warmup and reaches it at ``steps`` -- the RL recipes never
    # asked for that, so the default here is a *constant* LR after warmup
    # (``lr_decay_ratio=0.0``, the usual RL choice and what the Miles
    # baselines run). Set ``lr_decay_ratio`` in (0, 1] for a
    # warmup-stable-decay schedule over the last fraction of ``steps``, or
    # ``None`` for torchtitan's decay-immediately behaviour.
    lr_decay_ratio: float | None = 0.0
    lr_decay_type: Literal["linear", "sqrt", "cosine"] = "linear"
    lr_min_factor: float = 0.0
    dp_shard_degree: int = -1
    dp_replicate_degree: int = 1
    tp_degree: int = 1
    cp_degree: int = 1
    enable_checkpoint: bool = False
    checkpoint_folder: str = "checkpoint"
    dump_folder: str = "./outputs"
    compile_model: bool = False
    compile_backend: str = "inductor"
    # torchtitan's per-transformer-block activation checkpointing. "selective"
    # (per-op SAC) is torchtitan's default and fine at short context; long-
    # context recipes need "full" because the retained per-layer activations
    # scale with the CP-local sequence length.
    activation_checkpoint_mode: Literal[
        "selective", "full", "memory_budget", "none"
    ] = "selective"
    # Inner-attention kernel. "sdpa" is plain causal attention (required by
    # context parallelism and by the "padded" batch layout); "varlen" is
    # torch's variable-length flash kernel and is required by the "packed"
    # layout (see ``TrainerParamsConfig.batch_layout``).
    attn_backend: Literal["sdpa", "varlen"] = "sdpa"

    def __post_init__(self) -> None:
        if self.attn_backend not in ("sdpa", "varlen"):
            raise ValueError(
                f"attn_backend must be 'sdpa' or 'varlen', got {self.attn_backend!r}"
            )
        if self.lr_decay_ratio is not None and not (0.0 <= self.lr_decay_ratio <= 1.0):
            raise ValueError(
                f"lr_decay_ratio must be None or within [0, 1], got {self.lr_decay_ratio!r}"
            )
        if self.lr_decay_type not in ("linear", "sqrt", "cosine"):
            raise ValueError(
                f"lr_decay_type must be 'linear', 'sqrt' or 'cosine', got {self.lr_decay_type!r}"
            )
        if not (0.0 <= self.lr_min_factor <= 1.0):
            raise ValueError(f"lr_min_factor must be within [0, 1], got {self.lr_min_factor!r}")


@dataclass
class TrainerParamsConfig:
    mini_batch_size: int = 1
    micro_batch_size: int = 1
    # Dynamic batching. ``batch_layout`` picks how a micro-batch is laid out
    # for the forward: "padded" pads every sample to the micro-batch's
    # longest sample (rounded up to ``seq_bucket``), "packed" concatenates
    # the samples into one row with cu_seqlens (needs attn_backend="varlen",
    # no context parallelism). ``max_tokens_per_micro`` sizes micro-batches
    # by tokens instead of by ``micro_batch_size`` rows; it is mandatory for
    # "packed" and, when set, takes precedence over ``micro_batch_size``.
    batch_layout: Literal["padded", "packed"] = "padded"
    max_tokens_per_micro: int | None = None
    seq_bucket: int = 2048
    ppo_clip_eps_low: float = 0.2
    ppo_clip_eps_high: float = 0.2
    old_logprobs_source: Literal["rollout", "train"] = "rollout"
    calculate_per_token_loss: bool = False
    use_tis: bool = False
    # IcePop uses train-old / rollout weights; legacy TIS masks new / old.
    # Enabling this also freezes train-old logprobs across a complete window.
    use_icepop: bool = False
    positive_lm_loss_coef: float = 0.0
    tis_ratio_min: float = 0.5
    tis_ratio_max: float = 5.0
    logprob_chunk_size: int = 1024
    # Report the policy entropy over loss tokens (``train/entropy``). Costs
    # one extra no-grad softmax pass over the logits in ``entropy_chunk_size``
    # token slices (fp32 ``[rows, entropy_chunk_size, V]`` transient).
    log_entropy: bool = True
    entropy_chunk_size: int = 512
    # ── critic advantages (PPO with an independent value net) ────────────
    # With ``enable_gae`` the trainer builds per-token advantages itself from
    # the ``values`` column a critic Service published and the ``reward``
    # column the rollout shaped, instead of reading a ready-made per-sequence
    # ``advantage``. The recursion is the same one the critic used to run
    # (``meshy.backend.titan.critic.gae``); moving it here is what lets the
    # advantage stay per-token all the way into the PPO ratio.
    enable_gae: bool = False
    normalize_advantages: bool = False
    gae_gamma: float = 1.0
    # VAPO length-adaptive lambda: lambda_i = 1 - 1/(alpha * L_i). Ignored when
    # ``gae_lambda`` pins a constant instead.
    gae_alpha: float = 1.5
    gae_lambda: float | None = None

    def __post_init__(self) -> None:
        if self.entropy_chunk_size <= 0:
            raise ValueError("entropy_chunk_size must be positive")
        if self.use_icepop and self.old_logprobs_source != "train":
            raise ValueError("IcePop requires old_logprobs_source='train'")
        if self.positive_lm_loss_coef < 0:
            raise ValueError("positive_lm_loss_coef must be nonnegative")
        if self.normalize_advantages and not self.enable_gae:
            raise ValueError("normalize_advantages requires enable_gae")
        if self.positive_lm_loss_coef and self.enable_gae and self.gae_gamma != 1.0:
            raise ValueError("Positive LM reward gating requires undiscounted returns (gae_gamma=1)")
        if not 0.0 <= self.gae_gamma <= 1.0:
            raise ValueError(f"gae_gamma must be within [0, 1], got {self.gae_gamma!r}")
        if self.gae_alpha <= 0:
            raise ValueError("gae_alpha must be positive")
        if self.gae_lambda is not None and not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(
                f"gae_lambda must be None or within [0, 1], got {self.gae_lambda!r}"
            )
        if self.old_logprobs_source not in ("rollout", "train"):
            raise ValueError(
                "old_logprobs_source must be 'rollout' or 'train', "
                f"got {self.old_logprobs_source!r}"
            )
        if self.tis_ratio_min <= 0 or self.tis_ratio_max < self.tis_ratio_min:
            raise ValueError("TIS ratio bounds must satisfy 0 < min <= max")
        if self.logprob_chunk_size <= 0:
            raise ValueError("logprob_chunk_size must be positive")
        if self.batch_layout not in ("padded", "packed"):
            raise ValueError(
                f"batch_layout must be 'padded' or 'packed', got {self.batch_layout!r}"
            )
        if self.max_tokens_per_micro is not None and self.max_tokens_per_micro <= 0:
            raise ValueError("max_tokens_per_micro must be positive when set")
        if self.batch_layout == "packed" and self.max_tokens_per_micro is None:
            raise ValueError("batch_layout='packed' requires max_tokens_per_micro")
        if self.seq_bucket <= 0:
            raise ValueError("seq_bucket must be positive")


@dataclass
class ServiceConfig:
    role: ClassVar[str] = "?"
    service_cls: ClassVar[str] = ""
    uses_gpu: ClassVar[bool] = True
    endpoint_port_base: ClassVar[int | None] = None
    dist_port_base: ClassVar[int | None] = None


DEFAULT_DATA_PARTITION = "data.train"
# Every column the rollout writes per sample and the trainer fetches. The
# trainer's fetch is an AND-filter over these, so producer and consumer must
# agree; ``meshy.worker.rollout.GRPO_FIELDS`` re-exports this list.
GRPO_TRAINER_FIELDS = [
    "tokens",
    "logprobs",
    "mask_assistant",
    "advantage",
    "weight_version",
    # raw scalar reward -> ``rollout/raw_reward_mean`` and group statistics
    "reward",
    # 0/1 stamps -> ``rollout/truncated_ratio``, ``rollout/repetition_frac``,
    # ``rollout/weight_version/mixed_version_ratio``
    "truncated",
    "repetition",
    "mixed_version",
]


@dataclass
class InferenceServiceConfig(ServiceConfig):
    role: ClassVar[str] = "inference"
    service_cls: ClassVar[str] = "meshy.service.inference:SGLangService"
    endpoint_port_base: ClassVar[int] = 30000
    dist_port_base: ClassVar[int] = 40000

    model_path: str | None = None
    server_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingServiceConfig(ServiceConfig):
    role: ClassVar[str] = "training"
    service_cls: ClassVar[str] = "meshy.service.training:TitanTrainingService"
    endpoint_port_base: ClassVar[int] = 31000
    dist_port_base: ClassVar[int] = 41000

    model_path: str
    trainer_config: TrainerConfig
    batch_size: int
    trainer_params: TrainerParamsConfig | None = None
    timer_enabled: bool = True
    weight_sync_mode: Literal["disk", "auto"] = "auto"
    #: Exported HF weight versions (``weights/{service}/v{N}``) kept on disk.
    #: One full model is written *every* step for the inference hand-off, so
    #: without rotation a long run fills the volume -- 296 steps of a 2.25B
    #: model is terabytes. Versions a retained global checkpoint still points
    #: at are never pruned, whatever this says. 0 keeps everything.
    #:
    #: Keep this above the run's staleness window: a colocated critic hands
    #: the card back naming the version its *rows* were generated against
    #: (``CriticWorker._actor_weights_for``), which lags the current step by
    #: up to ``pacing_window`` -- or, when the rollout runs ungated, by
    #: however far its in-flight cap lets it run ahead.
    weights_keep_latest_k: int = 5
    stream_minibatch: bool = False
    tq_endpoints_file: str | None = None
    partition_id: str = DEFAULT_DATA_PARTITION
    tq_fields: list[str] = field(default_factory=lambda: list(GRPO_TRAINER_FIELDS))
    tq_poll_interval: float = 0.5
    #: Number of gen gates a critic Service raises during its cold start, while
    #: this trainer is idle. The trainer resumes gate numbering above them so
    #: the rollout's gate stream stays monotonic. Must match
    #: ``CriticServiceConfig.cold_start_windows``; 0 when there is no critic.
    critic_cold_start_windows: int = 0
    #: Global checkpoint / resume policy. Pass the *same* instance to the
    #: critic and the rollout; the topology rejects a mismatch.
    checkpoint: GlobalCheckpointConfig = field(default_factory=GlobalCheckpointConfig)


@dataclass
class RolloutServiceConfig(ServiceConfig):
    role: ClassVar[str] = "rollout"
    service_cls: ClassVar[str] = "meshy.service.rollout:RolloutService"
    uses_gpu: ClassVar[bool] = False

    model_path: str
    dataset: str
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    reward: str | Callable[[Any], float] | None = None
    #: Per-sample reward shaping applied to the raw task reward, e.g. the
    #: recipe's soft-overlong penalty
    #: (``meshy.reward:soft_overlong_penalty``). When set, ``reward`` on the
    #: wire is the shaped ``R`` the critic and the GAE recursion consume, and
    #: the unshaped value is preserved in the ``raw_reward`` column. Unlike
    #: ``advantage`` this is not skipped by ``external_advantage``: the
    #: shaping belongs to the reward, not to the advantage estimator.
    reward_shaping: str | Callable[..., float] | None = None
    reward_shaping_kwargs: dict[str, Any] = field(default_factory=dict)
    advantage: str | Callable[[list[Any]], Any] | None = None
    advantage_kwargs: dict[str, Any] = field(default_factory=dict)
    filter_zero_std_groups: bool = False
    publish_positive_lm_weights: bool = False
    #: Cap on how much a run may oversample to replace groups that
    #: ``filter_zero_std_groups`` dropped, as a multiple of the dataset. 1.0
    #: disables replacement (a dropped group is simply lost, shortening the
    #: epoch); 2.0 is the recipe's 2x oversampling budget. Inert when
    #: ``filter_zero_std_groups`` is False.
    oversample_factor: float = 1.0
    sampling_params: dict[str, Any] = field(default_factory=dict)
    group_size: int = 1
    num_epochs: int = 1
    poll_interval: float = 2.0
    async_max_running_request: int = -1
    pacing_window: int | str | None = "auto"
    tq_endpoints_file: str | None = None
    partition_id: str = DEFAULT_DATA_PARTITION
    verbose_trajectory_log: bool = False
    trajectory_log: str | None = None
    #: Global checkpoint / resume policy; see ``TrainingServiceConfig``.
    checkpoint: GlobalCheckpointConfig = field(default_factory=GlobalCheckpointConfig)
    #: Hand the advantage to a critic Service: the rollout stops computing it
    #: and stops writing the column, so the trainer's AND-filter holds each row
    #: back until the critic publishes one. Mutually exclusive with
    #: ``advantage``. See :class:`meshy.config.CriticServiceConfig`.
    external_advantage: bool = False

    def __post_init__(self) -> None:
        if self.oversample_factor < 1.0:
            raise ValueError(
                f"oversample_factor must be >= 1, got {self.oversample_factor!r}"
            )
        if self.reward_shaping is not None and self.reward is None:
            raise ValueError("reward_shaping needs a reward to shape")


# ── PPO critic (VAPO-GAE advantage) ─────────────────────────────────────
# The critic reads these columns off each rollout row and publishes the
# ``advantage`` the rollout deliberately left unwritten. As with OPD, TQ's
# AND-filter is what sequences rollout -> critic -> trainer; no participant
# needs to know about the others.
CRITIC_INPUT_FIELDS = ["tokens", "mask_assistant", "reward", "weight_version"]
#: What the critic writes back onto each rollout row, per ``publish_mode``.
#: ``"values"`` hands the trainer the per-token value function and lets it run
#: GAE (``TrainerParamsConfig.enable_gae``); ``"advantage"`` is the original
#: arrangement where the critic reduces to one scalar per sequence. Either way
#: it is the column the trainer's AND-filter waits on.
CRITIC_OUTPUT_FIELDS_BY_MODE = {
    "values": ["values"],
    "advantage": ["advantage"],
}
CRITIC_OUTPUT_FIELDS = CRITIC_OUTPUT_FIELDS_BY_MODE["advantage"]

#: Trainer columns when the critic publishes ``values`` and the trainer runs
#: GAE itself. ``advantage`` is gone (nobody writes it) and ``values`` takes
#: its place as the column that holds a row back until the critic has scored
#: it. ``reward`` is already in the GRPO list and is now load-bearing rather
#: than a metric: it carries the shaped ``R`` the GAE recursion consumes.
GAE_TRAINER_FIELDS = [
    field for field in GRPO_TRAINER_FIELDS if field != "advantage"
] + ["values"]

#: Add ``raw_reward`` when the rollout applies reward shaping
#: (``RolloutServiceConfig.reward_shaping``). It is *not* in the default list
#: because the trainer's fetch is an AND-filter: asking for a column no
#: producer writes would hold every row back forever. Shaping makes it
#: necessary rather than merely informative -- ``reward`` is then ``R``, and a
#: correct-but-overlong sample can carry ``R < 0.5``, so solve-rate metrics
#: have to read the unshaped ``r``.
GAE_TRAINER_FIELDS_SHAPED = GAE_TRAINER_FIELDS + ["raw_reward"]


@dataclass
class CriticServiceConfig(ServiceConfig):
    """An independent PPO critic replica that owns the advantage signal.

    A separate SPMD (FSDP) replica holding its own value net -- backbone from
    the base model, scalar value head initialised randomly -- sharing no
    parameter with the actor. Its only coupling to the actor is the
    ``advantage`` column it writes back to each rollout row.

    See ``justrl_ii_recipe.md`` §2 for the cold-start discipline the defaults
    encode, and ``docs/critic_engine.md`` for the value net itself.
    """

    role: ClassVar[str] = "critic"
    service_cls: ClassVar[str] = "meshy.service.critic:CriticService"
    endpoint_port_base: ClassVar[int] = 33000
    dist_port_base: ClassVar[int] = 43000

    #: base model the backbone is loaded from; the value head is never loaded
    #: and stays zero-initialised
    model_path: str
    #: reuses the trainer config for seq_len / dtype / parallel degrees
    trainer_config: TrainerConfig
    #: rows per critic window
    score_batch_size: int
    #: What the critic writes back per row. ``"values"`` publishes the
    #: per-token value function and leaves GAE to the trainer (which must set
    #: ``TrainerParamsConfig.enable_gae`` and fetch ``GAE_TRAINER_FIELDS``);
    #: ``"advantage"`` keeps the reduction inside the critic. See
    #: ``CRITIC_OUTPUT_FIELDS_BY_MODE``.
    publish_mode: Literal["values", "advantage"] = "values"
    #: Rows resident on the GPU per critic forward. This bounds memory only --
    #: the value step still accumulates over the whole window and takes one
    #: optimiser step (recipe §2 counts critic iterations in rollout steps).
    #: Ignored when ``max_tokens_per_micro`` is set, which is the form to
    #: prefer: a fixed row count has to be sized for the *longest* row in a
    #: window, and at 128k the rows span two orders of magnitude, so it spends
    #: most forwards on a nearly empty batch dimension.
    micro_rows: int = 1
    #: Token budget per critic forward (rows x padded row length). The planner
    #: sorts the window by length and packs to this, so short rows ride
    #: together and only genuinely long ones get a forward to themselves.
    #: ``None`` falls back to ``micro_rows``.
    max_tokens_per_micro: int | None = None
    #: Train the whole backbone, not just the value head. A value-head-only
    #: critic is a linear probe on frozen features and measurably plateaus
    #: before it beats the whitening baseline (var_reduction stays < 0).
    train_backbone: bool = True
    lr: float = 5e-6
    optimizer_kwargs: dict[str, Any] | None = None
    value_clip: float | None = None
    #: critic lr warmup, recipe §2
    warmup_steps: int = 10
    #: Windows the critic consumes on its own before publishing any advantage,
    #: keeping the actor frozen while the cold-start value-loss spike is
    #: absorbed (recipe §2). The critic paces the rollout itself during these.
    #: -1 keeps the critic in cold start indefinitely, clearing every window.
    cold_start_windows: int = 30
    #: Publish the initial generation gate in runs without an actor trainer.
    publish_gate_zero: bool = False
    #: passes over each window's value targets
    value_epochs: int = 1
    #: return discount; 1.0 for outcome-reward-only tasks
    gamma: float = 1.0
    #: VAPO length-adaptive lambda, lambda_i = 1 - 1/(alpha * L_i). Recipe §1:
    #: the paper's 0.05 collapses at 128k -- alpha must match response length.
    alpha: float = 1.5
    #: value-loss gradient clipping
    max_norm: float = 1.0
    timer_enabled: bool = True
    tq_endpoints_file: str | None = None
    partition_id: str = DEFAULT_DATA_PARTITION
    tq_poll_interval: float = 0.5
    #: Global checkpoint / resume policy; see ``TrainingServiceConfig``. The
    #: critic is the one role with *no* other persistence: without this a crash
    #: loses the value net and the whole cold start has to be paid again.
    checkpoint: GlobalCheckpointConfig = field(default_factory=GlobalCheckpointConfig)

    def __post_init__(self) -> None:
        if self.publish_mode not in CRITIC_OUTPUT_FIELDS_BY_MODE:
            raise ValueError(
                f"publish_mode must be one of "
                f"{sorted(CRITIC_OUTPUT_FIELDS_BY_MODE)}, got {self.publish_mode!r}"
            )
        if self.micro_rows <= 0:
            raise ValueError("micro_rows must be positive")

    @property
    def output_fields(self) -> list[str]:
        return list(CRITIC_OUTPUT_FIELDS_BY_MODE[self.publish_mode])


# ── Student Top-K on-policy distillation (OPD) ──────────────────────────
# Columns the Teacher appends to each rollout row: per position ``[L, K]``
# token ids and their Teacher log-probs, indexed by *logits position* (row
# ``t`` describes token ``t + 1``) so they line up with the trainer's
# ``labels`` without another shift. The Student fetches the GRPO columns
# plus these two; TQ's AND-filter is what sequences rollout -> Teacher ->
# Student, so neither the rollout nor the inference service knows about OPD.
OPD_TEACHER_FIELDS = ["teacher_topk_ids", "teacher_topk_logprobs"]
OPD_TRAINER_FIELDS = GRPO_TRAINER_FIELDS + OPD_TEACHER_FIELDS


@dataclass
class OPDTeacherConfig(ServiceConfig):
    """A Titan-hosted Teacher that scores rollout rows with Top-K log-probs.

    The Teacher is an SPMD (FSDP) replica like the trainer, so it can sit in a
    colocation ring next to the Student trainer and the inference server.
    """

    role: ClassVar[str] = "teacher"
    service_cls: ClassVar[str] = "meshy.service.opd:OPDTeacherService"
    endpoint_port_base: ClassVar[int] = 32000
    dist_port_base: ClassVar[int] = 42000

    model_path: str
    trainer_config: TrainerConfig
    #: number of Teacher candidates kept per position
    top_k: int = 8
    #: rows fetched and scored per Teacher forward window; the training
    #: ``batch_size`` should be a multiple of it
    score_batch_size: int = 8
    #: name of the student training group whose checkpoints the inference
    #: server must reload when the GPU returns to it after a Teacher window
    #: (``None`` = the first training service in the topology)
    student_group: str | None = None
    tq_endpoints_file: str | None = None
    partition_id: str = DEFAULT_DATA_PARTITION
    tq_poll_interval: float = 0.5
    timer_enabled: bool = False

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("OPDTeacherConfig.top_k must be positive")
        if self.score_batch_size <= 0:
            raise ValueError("OPDTeacherConfig.score_batch_size must be positive")
        if self.trainer_config.cp_degree != 1:
            raise ValueError("the OPD Teacher does not support context parallelism yet")


@dataclass
class OPDTrainingConfig(TrainingServiceConfig):
    """Student trainer: the ordinary Titan training service with the Top-K loss.

    Everything the base config offers (parallelism, dynamic batching,
    colocation, weight sync) is inherited; only the loss and the fetched
    columns differ.
    """

    service_cls: ClassVar[str] = "meshy.service.opd:OPDTrainingService"

    tq_fields: list[str] = field(default_factory=lambda: list(OPD_TRAINER_FIELDS))
    #: clamp student/teacher Top-K log-probs from below before the KL
    #: (VERL default -10; ``None`` disables)
    log_prob_min_clamp: float | None = -10.0
    #: clamp the per-token KL from above (VERL default 10; ``None`` disables)
    loss_max_clamp: float | None = 10.0


__all__ = [
    "CRITIC_INPUT_FIELDS",
    "CRITIC_OUTPUT_FIELDS",
    "CRITIC_OUTPUT_FIELDS_BY_MODE",
    "CriticServiceConfig",
    "DEFAULT_DATA_PARTITION",
    "GAE_TRAINER_FIELDS",
    "GAE_TRAINER_FIELDS_SHAPED",
    "GRPO_TRAINER_FIELDS",
    "GlobalCheckpointConfig",
    "InferenceServiceConfig",
    "OPDTeacherConfig",
    "OPDTrainingConfig",
    "OPD_TEACHER_FIELDS",
    "OPD_TRAINER_FIELDS",
    "RolloutServiceConfig",
    "SamplingParams",
    "ServiceConfig",
    "TrainerConfig",
    "TrainerParamsConfig",
    "TrainingServiceConfig",
]
