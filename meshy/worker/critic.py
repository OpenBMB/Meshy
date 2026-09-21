"""Critic Worker: score rollout rows, publish the value function, train it.

The critic is the actor's counterpart, not an attachment to it. It owns an
independent value network and its only coupling to the actor is the one column
it writes onto each rollout row.

Sequencing is done entirely by TransferQueue's AND-filter, exactly as OPD
sequences rollout -> Teacher -> Student. The rollout deliberately leaves that
column unwritten (``RolloutWorker(external_advantage=True)``), so a row is
invisible to the trainer until the critic fills it in::

    rollout  writes GRPO_FIELDS_NO_ADVANTAGE (reward = the shaped R)
    critic   reads  CRITIC_INPUT_FIELDS, writes "values"
    trainer  fetches GAE_TRAINER_FIELDS -> blocked until the critic wrote it,
             then runs GAE over (values, reward) itself

``publish_mode="advantage"`` keeps the older arrangement, where the critic runs
GAE and publishes one scalar per sequence into ``advantage`` and the trainer
fetches ``GRPO_TRAINER_FIELDS``. The pipeline shape is identical either way;
only the column and who runs the recursion differ.

Order within a window is load-bearing: **score with the current value function,
publish, and only then update it**. Training first would score the actor's data
with a value function already fitted to that same data, biasing the advantage
the actor consumes.

Cold start (``justrl_ii_recipe.md`` §2): for the first ``cold_start_windows``
the critic trains on rows but publishes nothing and clears them itself, so the
trainer never sees a row and the actor genuinely never steps while the
cold-start value-loss spike is absorbed. During that phase the critic also
raises the gen gate -- it is the only consumer of rollout output, so it is the
only thing that can pace the rollout (see :meth:`process_tq_batch`).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from loguru import logger

from meshy.config import CRITIC_INPUT_FIELDS, CRITIC_OUTPUT_FIELDS_BY_MODE
from meshy.transferqueue.control import (
    GEN_GATE_FIELDS,
    GEN_GATE_PARTITION,
)
from meshy.worker.tq import TQInput, TQOutput, TQWorker


class CriticWorker(TQWorker):
    """Consume reward-bearing rollout rows; emit advantages and train the critic.

    The rows stay in the partition (``clear_after_success=False``) once the
    critic is publishing, because the trainer consumes and clears them. During
    cold start that flips: nothing downstream will ever read those rows, so the
    critic clears them itself (:meth:`should_clear_batch`).
    """

    def __init__(
        self,
        *,
        engine: Any,
        endpoints_ref: str,
        partition_id: str,
        score_batch_size: int,
        cold_start_windows: int = 0,
        publish_mode: str = "values",
        publish_gate_zero: bool = False,
        poll_interval: float = 0.5,
        colocation: Any = None,
        actor_weights: Callable[[int], str | None] | None = None,
        resume_step: int = 0,
        client_factory=None,
    ) -> None:
        super().__init__()
        if score_batch_size <= 0:
            raise ValueError("score_batch_size must be positive")
        if cold_start_windows < -1:
            raise ValueError("cold_start_windows must be -1 (infinite) or >= 0")
        if publish_mode not in CRITIC_OUTPUT_FIELDS_BY_MODE:
            raise ValueError(
                f"publish_mode must be one of "
                f"{sorted(CRITIC_OUTPUT_FIELDS_BY_MODE)}, got {publish_mode!r}"
            )
        #: The column this critic owns; also the name of the engine result key
        #: and of the declared TQ output, so the three cannot drift.
        self.publish_mode = publish_mode
        self.output_field = CRITIC_OUTPUT_FIELDS_BY_MODE[publish_mode][0]
        self.engine = engine
        self.colocation = colocation
        #: ``(max weight_version in the window) -> actor checkpoint path``. The
        #: critic trains nothing the inference server serves, so when the ring
        #: hands the card back to SGLang it must name the weights the card
        #: already held; see :meth:`process_tq_batch`.
        self.actor_weights = actor_weights
        self.cold_start_windows = int(cold_start_windows)
        self.publish_gate_zero = bool(publish_gate_zero)
        #: Windows consumed, and the run's global clock: it counts from the
        #: first window, cold start included, which is exactly the gate step
        #: the trainer's ``gate_step_offset`` is defined against. Restoring it
        #: is load-bearing beyond bookkeeping -- :meth:`should_clear_batch`
        #: reads it to decide whether a window's rows are the critic's to drop
        #: or the trainer's to consume, so a reset counter makes the critic
        #: swallow rows the actor should have trained on.
        self.windows = int(resume_step)
        #: gates raised during cold start; the trainer resumes numbering above
        #: this via ``TitanWorker.gate_step_offset``. On resume the trainer's
        #: genesis pulse re-establishes the stream at ``resume_step``, so the
        #: next cold-start gate this worker raises must be the one after it.
        self.gates_raised = int(resume_step)
        self.configure_tq(
            endpoints_ref=endpoints_ref,
            input=TQInput(
                partition=partition_id,
                fields=tuple(CRITIC_INPUT_FIELDS),
                batch_size=int(score_batch_size),
                consumer="critic",
                clear_after_success=False,
            ),
            outputs={
                self.output_field: TQOutput(
                    fields=(self.output_field,), new_rows=False
                ),
                "gate": TQOutput(
                    fields=tuple(GEN_GATE_FIELDS),
                    new_rows=True,
                    partition=GEN_GATE_PARTITION,
                ),
            },
            poll_interval=poll_interval,
            client_factory=client_factory,
        )

    @property
    def in_cold_start(self) -> bool:
        """Whether the *next* window is still a cold-start window."""
        return self.cold_start_windows < 0 or self.windows < self.cold_start_windows

    def should_clear_batch(self) -> bool:
        """Cold-start rows are the critic's to drop; later rows belong to the trainer.

        ``self.windows`` has already been incremented by the time TQ asks, so
        this reports on the window just processed.
        """
        return self.cold_start_windows < 0 or self.windows <= self.cold_start_windows

    def startup_tq_outputs(self) -> Mapping[str, Any]:
        """Release rollout before any rows exist when no trainer supplies gate 0."""
        if not self.publish_gate_zero:
            return {}
        from meshy.transferqueue.control import make_gen_gate

        logger.info("Critic raised gate 0 (v0); no trainer in this run")
        return {"gate": make_gen_gate(step=0, weight_version=0)}

    def process_tq_batch(self, samples: list[Any]) -> Mapping[str, Any]:
        from meshy.transferqueue import adapter

        cold = self.in_cold_start
        self.windows += 1
        request = None
        if self.colocation is not None:
            request = self.colocation.request_gpu(
                request_id=f"{getattr(self.engine, 'name', 'critic')}:window:{self.windows}"
            )
            self.colocation.wait_for_grant(request)
        try:
            result = self.engine.score_and_train(
                samples, publish=not cold, global_step=self.windows
            )
        finally:
            if request is not None:
                # The next ring member is the trainer, but it cannot have asked
                # for the card yet: this window's advantage column is written
                # only after ``process_tq_batch`` returns, so the rows are still
                # invisible to it. The ring therefore falls back to the
                # inference server, whose acquire callback reloads whatever the
                # grant names -- and a grant without a path is fatal there
                # (``SGLangEngine.on_colocate_acquire``). The critic trained no
                # actor weights, so it names the ones the rows were generated
                # against.
                self.colocation.release(
                    transition="critic-window-complete",
                    payload_ref=self._actor_weights_for(samples),
                )
        if result is None:
            raise RuntimeError("CriticEngine.score_and_train returned no result on the Worker rank")

        self._log_window(result, cold=cold)

        if cold:
            # Nothing downstream will read these rows, so the critic is also
            # the only thing pacing the rollout: without a gate here the
            # rollout stalls at its pacing window while the trainer sits idle.
            self.gates_raised += 1
            return {"gate": self._gate(self.gates_raised)}

        scored = result[self.output_field]
        if len(scored) != len(samples):
            raise ValueError(
                f"critic produced {len(scored)} {self.output_field} rows for "
                f"{len(samples)} input rows"
            )
        return {
            self.output_field: adapter.samples_to_td(scored, (self.output_field,))
        }

    def _actor_weights_for(self, samples: list[Any]) -> str | None:
        """Checkpoint the inference server must hold after this critic window."""
        if self.actor_weights is None:
            return None
        version = max(int(td["weight_version"]) for td in samples)
        return self.actor_weights(version)

    def _gate(self, step: int) -> Any:
        from meshy.transferqueue.control import make_gen_gate

        version = int(getattr(self.engine, "weight_version", 0))
        logger.info(
            "Critic raised cold-start gen gate {} (v{}); actor stays frozen",
            step,
            version,
        )
        return make_gen_gate(step=step, weight_version=version)

    def _log_window(self, result: Mapping[str, Any], *, cold: bool) -> None:
        limit = "inf" if self.cold_start_windows < 0 else str(self.cold_start_windows)
        phase = (
            f"cold start {self.windows}/{limit}"
            if cold
            else f"window {self.windows}"
        )
        diag = result.get("diagnostics")
        logger.info(
            "Critic {}: value_loss={:.4f} rows={} | {}",
            phase,
            float(result.get("value_loss", float("nan"))),
            int(result.get("rows", 0)),
            diag if diag is not None else "(no gauges)",
        )

    def tq_health_info(self) -> dict[str, Any]:
        info = super().tq_health_info()
        info["critic"] = {
            "windows": self.windows,
            "cold_start_windows": self.cold_start_windows,
            "in_cold_start": self.in_cold_start,
            "gates_raised": self.gates_raised,
        }
        return info


__all__ = ["CriticWorker"]
