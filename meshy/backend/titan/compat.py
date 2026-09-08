"""Compatibility shims for torchtitan's ``experiments.forge`` engine.

Upstream torchtitan refactored its loss plumbing: the loss is no longer a
``ModelSpec`` callable (``build_loss_fn=build_cross_entropy_loss``) but a
``BaseLoss.Config`` living on the *job* config, built via
``config.loss.build(compile_config=...)`` (see ``torchtitan/trainer.py``).
``experiments/forge/engine.py`` was not carried along in that refactor, so on
the pinned nightly (``torchtitan 0.1.0.dev20260501``) ``ForgeEngine.__init__``
carries two stale references that fire on *every* construction:

1. ``self.loss_fn = self.train_spec.loss.build(config.compile,
   parallel_dims=parallel_dims)`` -- ``ModelSpec`` has no ``loss`` field any
   more (``AttributeError``), and ``Configurable.Config.build`` is
   keyword-only, so the positional ``config.compile`` plus the unknown
   ``parallel_dims`` kwarg would fail as well (``TypeError``).
2. ``parallelism_config.disable_loss_parallel`` -- nothing in the function
   binds ``parallelism_config``, so the name resolves against the *module*
   globals and raises ``NameError``.

Rather than fork ~150 lines of engine setup into this repo (which would then
rot against every nightly), we feed the engine what those two lines ask for:

* a duck-typed loss config attached to the ``ModelSpec`` instance, tolerant of
  the legacy ``build()`` calling convention, producing the same plain
  cross-entropy loss the old ``build_cross_entropy_loss`` did;
* a module-global ``parallelism_config`` in ``torchtitan.experiments.forge
  .engine``, which is what an unbound name in that function falls back to.

Both are inert once upstream fixes the engine: a real ``ModelSpec.loss`` field
is left untouched, and a properly bound local ``parallelism_config`` shadows
the module global.

Note that the loss produced here is a placeholder as far as this repo is
concerned: :class:`~meshy.backend.titan.trainer.TitanTrainer` computes its own
PPO objective and never calls ``self.loss_fn``. It is only handed to
``pipelining_fn`` when pipeline parallelism is enabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchtitan.experiments.forge import ForgeEngine

__all__ = ["apply_forge_engine_compat"]


class _CrossEntropyLossSpec:
    """Stand-in for the ``ModelSpec.loss`` config the forge engine expects.

    Accepts the compile config either positionally (how the stale engine calls
    it) or by keyword, and swallows extra kwargs such as ``parallel_dims`` that
    the current ``BaseLoss`` constructors no longer take.
    """

    def build(self, compile_config: Any = None, /, **kwargs: Any):
        from torchtitan.components.loss import CrossEntropyLoss

        config = CrossEntropyLoss.Config()
        if compile_config is None:
            compile_config = kwargs.get("compile_config")
        if compile_config is None:
            return config.build()
        return config.build(compile_config=compile_config)


def apply_forge_engine_compat(config: "ForgeEngine.Config") -> None:
    """Make *config* survive ``ForgeEngine.__init__`` on the pinned nightly.

    Idempotent, and safe to call once per process before each engine build.
    """
    model_spec = config.model_spec
    if getattr(model_spec, "loss", None) is None:
        # ``ModelSpec`` is a plain (non-slots) dataclass, so an out-of-band
        # attribute sticks; guard anyway so a future slots=True upstream
        # surfaces as a readable error instead of a bare AttributeError.
        try:
            model_spec.loss = _CrossEntropyLossSpec()
        except AttributeError as exc:  # pragma: no cover - upstream change
            raise RuntimeError(
                "Cannot attach a loss config to torchtitan's ModelSpec; the "
                "ForgeEngine loss shim in meshy.backend.titan.compat needs "
                "updating for this torchtitan version."
            ) from exc

    # ``ForgeEngine.__init__`` reads a free variable ``parallelism_config``
    # when it decides whether loss parallel is enabled; free names resolve
    # against the defining module's globals. One engine is built per process
    # (card-level SPMD), so a module global is unambiguous here.
    from torchtitan.experiments.forge import engine as _forge_engine

    _forge_engine.parallelism_config = config.parallelism
