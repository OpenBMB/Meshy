"""TransferQueue-backed Worker for :class:`meshy.engine.titan.TitanEngine`."""

from __future__ import annotations

from typing import Any, Mapping

from loguru import logger

from meshy.engine.titan import TitanEngine
from meshy.worker.tq import TQInput, TQOutput, TQWorker
from meshy.service.colocation import ColocationManager


class TitanWorker(TQWorker):
    """Consume rollout samples, publish checkpoints, and turn steps into gates.

    For a colocated trainer, the completed checkpoint path is attached to the
    next GPU grant.  The SGLang acquire callback then reloads exactly that
    checkpoint before generation resumes.
    """

    def __init__(
        self,
        *,
        engine: TitanEngine,
        endpoints_ref: str,
        partition_id: str,
        tq_fields: list[str] | tuple[str, ...],
        batch_size: int,
        poll_interval: float = 0.5,
        fetch_batch_size: int | None = None,
        stream_minibatch: bool | None = None,
        max_retries: int = 3,
        process_retries: int = 0,
        retry_interval: float = 1.0,
        client_factory=None,
        colocation: ColocationManager | None = None,
        gate_step_offset: int = 0,
        resume_step: int = 0,
    ) -> None:
        super().__init__()
        if not tq_fields:
            raise ValueError("TitanWorker requires non-empty tq_fields")
        if batch_size <= 0:
            raise ValueError("TitanWorker batch_size must be positive")
        if gate_step_offset < 0:
            raise ValueError("TitanWorker gate_step_offset must be >= 0")
        # A critic cold start paces the rollout itself for its first N windows
        # (the trainer is idle, so it cannot). Those gates occupy steps
        # ``1..N``, so the trainer resumes numbering above them and the gate
        # stream the rollout reads stays monotonic across the hand-off.
        self.gate_step_offset = int(gate_step_offset)
        #: Global window index a resumed run re-establishes the gate stream at.
        self._resume_step = int(resume_step)
        self.engine = engine
        self.colocation = colocation
        self._colocation_request = None
        self.batch_size = int(batch_size)
        self.fetch_batch_size = int(fetch_batch_size or batch_size)
        if self.fetch_batch_size <= 0:
            raise ValueError("TitanWorker fetch_batch_size must be positive")
        self.stream_minibatch = bool(
            getattr(engine, "stream_minibatch", False)
            if stream_minibatch is None
            else stream_minibatch
        )
        # Nothing to restore on resume: a global checkpoint is only ever taken
        # on a ``sync`` step, where this counter has just been zeroed, so a
        # resumed worker always starts a window cleanly rather than
        # mid-accumulation.
        self.trained_since_sync = 0
        gen_gate_fields = ("gate_step", "weight_version")
        gen_gate_partition = "gen_gate"

        self.configure_tq(
            endpoints_ref=endpoints_ref,
            input=TQInput(
                partition=partition_id,
                fields=tuple(tq_fields),
                batch_size=self.fetch_batch_size,
                consumer="titan",
                clear_after_success=True,
            ),
            outputs={
                "gate": TQOutput(
                    fields=gen_gate_fields,
                    new_rows=True,
                    partition=gen_gate_partition,
                )
            },
            poll_interval=poll_interval,
            max_retries=max_retries,
            process_retries=process_retries,
            retry_interval=retry_interval,
            client_factory=client_factory,
        )

    def startup_tq_outputs(self) -> Mapping[str, Any]:
        # The genesis gate establishes the version inference may generate
        # against. It precedes any critic cold-start gate, so it is never
        # offset -- the rollout must see step 0 first.
        #
        # A resumed run starts with an empty ``gen_gate`` partition (TQ is
        # RAM-only and dies with the launcher), so the same pulse has to
        # re-establish the stream *at the step the run reached*. The rollout
        # restores ``gates_seen`` to that step and so expects exactly this
        # gate; numbering it 0 again would rewind every consumer's budget.
        step = self._resume_step if self._resume_step > 0 else self._current_version()
        return self._gate_output(step=step, offset=0)

    def _current_version(self) -> int:
        return int(getattr(self.engine, "weight_version", getattr(self.engine, "step_index", 0)))

    def _gate_output(self, *, step: int, offset: int | None = None) -> Mapping[str, Any]:
        from meshy.transferqueue.control import make_gen_gate

        version = self._current_version()
        step = int(step) + (self.gate_step_offset if offset is None else int(offset))
        logger.info("Titan {} raised gen gate {} (v{})", getattr(self.engine, "name", "titan"), step, version)
        return {"gate": make_gen_gate(step=step, weight_version=version)}

    def process_tq_batch(self, samples: list[Any]) -> Mapping[str, Any]:
        if self.colocation is not None and self._colocation_request is None:
            request_id = f"{getattr(self.engine, 'name', 'titan')}:window:{getattr(self.engine, 'step_index', 0)}"
            self._colocation_request = self.colocation.request_gpu(request_id=request_id)
            self.colocation.wait_for_grant(self._colocation_request)
        self.trained_since_sync += len(samples)
        sync = not self.stream_minibatch or self.trained_since_sync >= self.batch_size
        if sync:
            self.trained_since_sync = 0

        result = self.engine.step(samples, sync=sync)
        if not sync:
            return {}
        if result is None:
            raise RuntimeError("TitanEngine.step returned no result on the Worker rank")
        step = int(getattr(result, "step", self._current_version()))
        if self.colocation is not None:
            weights_path = getattr(result, "weights_path", None)
            if not weights_path:
                raise RuntimeError(
                    "colocated Titan step completed without a checkpoint path; "
                    "SGLang cannot be resumed with the latest weights"
                )
            self.colocation.release(
                transition="train-step-complete",
                payload_ref=weights_path,
            )
            self._colocation_request = None
        return self._gate_output(step=step)


__all__ = ["TitanWorker"]
