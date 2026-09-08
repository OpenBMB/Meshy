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
    tis_ratio_min: float = 0.5
    tis_ratio_max: float = 5.0
    logprob_chunk_size: int = 1024
    # Report the policy entropy over loss tokens (``train/entropy``). Costs
    # one extra no-grad softmax pass over the logits in ``entropy_chunk_size``
    # token slices (fp32 ``[rows, entropy_chunk_size, V]`` transient).
    log_entropy: bool = True
    entropy_chunk_size: int = 512

    def __post_init__(self) -> None:
        if self.entropy_chunk_size <= 0:
            raise ValueError("entropy_chunk_size must be positive")
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
    hf_save_interval: int = 0
    stream_minibatch: bool = False
    tq_endpoints_file: str | None = None
    partition_id: str = DEFAULT_DATA_PARTITION
    tq_fields: list[str] = field(default_factory=lambda: list(GRPO_TRAINER_FIELDS))
    tq_poll_interval: float = 0.5


@dataclass
class RolloutServiceConfig(ServiceConfig):
    role: ClassVar[str] = "rollout"
    service_cls: ClassVar[str] = "meshy.service.rollout:RolloutService"
    uses_gpu: ClassVar[bool] = False

    model_path: str
    dataset: str
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    reward: str | Callable[[Any], float] | None = None
    advantage: str | Callable[[list[Any]], Any] | None = None
    advantage_kwargs: dict[str, Any] = field(default_factory=dict)
    filter_zero_std_groups: bool = False
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
    "DEFAULT_DATA_PARTITION",
    "GRPO_TRAINER_FIELDS",
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
