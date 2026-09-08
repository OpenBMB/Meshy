"""Teacher Worker: fetch rollout rows, score them, append the Teacher columns."""

from __future__ import annotations

from typing import Any, Mapping

from loguru import logger

from meshy.worker.tq import TQInput, TQOutput, TQWorker

from meshy.config import OPD_TEACHER_FIELDS

#: Columns the Teacher reads from each rollout row.
TEACHER_INPUT_FIELDS = ("tokens", "mask_assistant", "weight_version")


class OPDTeacherWorker(TQWorker):
    """Consume rollout rows in ``score_batch_size`` windows and add Top-K columns.

    The rows stay in the partition (``clear_after_success=False``); the
    Student trainer, whose fetch requires the Teacher columns too, consumes and
    clears them afterwards.
    """

    def __init__(
        self,
        *,
        engine: Any,
        endpoints_ref: str,
        partition_id: str,
        score_batch_size: int,
        poll_interval: float = 0.5,
        colocation: Any = None,
        student_weights: Any = None,
        client_factory=None,
    ) -> None:
        super().__init__()
        if score_batch_size <= 0:
            raise ValueError("score_batch_size must be positive")
        self.engine = engine
        self.colocation = colocation
        # ``(max weight_version in the window) -> checkpoint path`` of the
        # student inference server; see :meth:`process_tq_batch`.
        self.student_weights = student_weights
        self.windows = 0
        self.configure_tq(
            endpoints_ref=endpoints_ref,
            input=TQInput(
                partition=partition_id,
                fields=TEACHER_INPUT_FIELDS,
                batch_size=int(score_batch_size),
                consumer="opd_teacher",
                clear_after_success=False,
            ),
            outputs={"teacher": TQOutput(fields=tuple(OPD_TEACHER_FIELDS), new_rows=False)},
            poll_interval=poll_interval,
            client_factory=client_factory,
        )

    def process_tq_batch(self, samples: list[Any]) -> Mapping[str, Any]:
        from tensordict import TensorDict

        from meshy.transferqueue import adapter

        self.windows += 1
        request = None
        if self.colocation is not None:
            request = self.colocation.request_gpu(
                request_id=f"{getattr(self.engine, 'name', 'teacher')}:score:{self.windows}"
            )
            self.colocation.wait_for_grant(request)
        try:
            scored = self.engine.score(samples)
        finally:
            if request is not None:
                # The ring hands the card back to the inference server, whose
                # acquire callback reloads the checkpoint named in the grant.
                # The Teacher trained nothing, so it names the student weights
                # the rollout rows were generated with.
                version = max(int(td["weight_version"]) for td in samples)
                payload = self.student_weights(version) if self.student_weights else None
                self.colocation.release(transition="teacher-score-complete", payload_ref=payload)
        if scored is None:
            raise RuntimeError("Teacher score returned no result on the Worker rank")
        logger.info("Teacher scored window {} ({} rows)", self.windows, len(scored))
        rows = [TensorDict(dict(row), batch_size=[]) for row in scored]
        return {"teacher": adapter.samples_to_td(rows, OPD_TEACHER_FIELDS)}


__all__ = ["OPDTeacherWorker", "TEACHER_INPUT_FIELDS"]
