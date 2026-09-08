"""ForgeEngine.Config builder for the TitanTrainer.

The job here is purely declarative: resolve a (model_name, model_flavor) pair
into a torchtitan ``ModelSpec`` and bundle it with the optimizer / scheduler /
parallelism / checkpoint sub-configs that ``ForgeEngine.__init__`` consumes.

No training-loop knobs (PPO clip, mini-batch size, ...) live here — those are
TitanTrainer ctor arguments, kept separate so the same forge config can be
reused by other trainers (SFT, eval, ...).

The flat field schema and per-field documentation live on
:class:`meshy.config.TrainerConfig`; this module only translates it into the
nested torchtitan config objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.forge import ForgeEngine

if TYPE_CHECKING:
    from meshy.config import TrainerConfig


def _get_model_spec(model_name: str, model_flavor: str, attn_backend: str = "sdpa"):
    """Resolve a (model_name, model_flavor) pair into a ModelSpec.

    Out-of-tree specs (qwen2_5, qwen2_5_math, minicpm5) live under
    :mod:`meshy.backend.titan.models` so the new trainer is self-contained.
    Every registry takes torchtitan's ``attn_backend`` name
    (``"sdpa" | "varlen" | "flex" | "flex_flash"``).
    """
    if model_name == "qwen3":
        from torchtitan.models.qwen3 import model_registry
        return model_registry(model_flavor, attn_backend=attn_backend)
    if model_name == "llama3":
        from torchtitan.models.llama3 import model_registry
        return model_registry(model_flavor, attn_backend=attn_backend)
    if model_name == "minicpm5":
        from .models.minicpm5 import model_registry
        return model_registry(model_flavor, attn_backend=attn_backend)
    if model_name == "qwen2_5":
        from .models.qwen2_5 import model_registry
        return model_registry(model_flavor, attn_backend=attn_backend)
    if model_name == "qwen2_5_math":
        from .models.qwen2_5_math import model_registry
        return model_registry(model_flavor, attn_backend=attn_backend)
    raise ValueError(
        f"Unsupported model_name '{model_name}'. Supported values: "
        "'qwen3', 'qwen2_5', 'qwen2_5_math', 'llama3', 'minicpm5'."
    )


def build_forge_config(
    trainer: "TrainerConfig",
    hf_model_path: str | None = None,
) -> ForgeEngine.Config:
    """Translate a :class:`~meshy.config.TrainerConfig` into a ForgeEngine.Config.

    Args:
        trainer: flat declarative trainer config (see
            :class:`meshy.config.TrainerConfig` for per-field documentation).
        hf_model_path: local path to HuggingFace model weights. When set, the
            checkpoint manager initialises from HF format.
    """
    model_spec = _get_model_spec(
        trainer.model_name, trainer.model_flavor, trainer.attn_backend
    )
    return ForgeEngine.Config(
        dump_folder=trainer.dump_folder,
        hf_assets_path=hf_model_path or "",
        model_spec=model_spec,
        optimizer=OptimizersContainer.Config(
            lr=trainer.lr,
            weight_decay=trainer.weight_decay,
            beta1=trainer.beta1,
            beta2=trainer.beta2,
            eps=trainer.eps,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=trainer.warmup_steps,
            # ``decay_ratio=0.0`` -> zero decay steps -> constant LR after
            # warmup (see ``TrainerConfig.lr_decay_ratio``).
            decay_ratio=trainer.lr_decay_ratio,
            decay_type=trainer.lr_decay_type,
            min_lr_factor=trainer.lr_min_factor,
        ),
        training=TrainingConfig(
            seq_len=trainer.seq_len,
            local_batch_size=trainer.local_batch_size,
            global_batch_size=trainer.global_batch_size,
            max_norm=trainer.max_norm,
            steps=trainer.steps,
            dtype=trainer.dtype,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=trainer.dp_shard_degree,
            data_parallel_replicate_degree=trainer.dp_replicate_degree,
            tensor_parallel_degree=trainer.tp_degree,
            context_parallel_degree=trainer.cp_degree,
        ),
        checkpoint=CheckpointManager.Config(
            # The CheckpointManager must be *enabled* whenever an initial HF
            # load is requested: ``CheckpointManager.load()`` starts with
            # ``if not self.enable: return False`` and never reaches the
            # ``initial_load_path`` branch, so ``enable_checkpoint=False`` +
            # ``hf_model_path`` would silently train a randomly initialized
            # model (all-ones RMSNorms, std-0.02 projections) while dumping
            # checkpoints with perfectly valid-looking keys/shapes.
            # ``trainer.enable_checkpoint`` keeps its user-facing meaning
            # (persist DCP checkpoints during training); the initial load is
            # enabled independently of it.
            enable=trainer.enable_checkpoint or bool(hf_model_path),
            # This is TorchTitan's native DCP resume directory. It is
            # intentionally independent from XRL_CHECKPOINT_DIR, which is
            # reserved for exported HF weights consumed by inference.
            folder=trainer.checkpoint_folder,
            initial_load_path=hf_model_path,
            initial_load_in_hf=bool(hf_model_path),
            initial_load_model_only=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode=trainer.activation_checkpoint_mode,
        ),
        compile=CompileConfig(
            enable=trainer.compile_model,
            components=["model"] if trainer.compile_model else [],
            backend=trainer.compile_backend,
        ),
        comm=CommConfig(),
        debug=DebugConfig(),
    )
