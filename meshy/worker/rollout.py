"""Gate-driven GRPO rollout Worker."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import math
import os
import time
from typing import Any, Callable

from loguru import logger

from meshy.config import GRPO_TRAINER_FIELDS
from meshy.engine.sglang import Generation, SGLangEngine
from meshy.utils.metric import has_repetition
from meshy.worker.tq import TQInput, TQOutput, TQWorker
from meshy.service.colocation import ColocationManager, RequestKind

#: Columns written per sample. Must equal what the trainer fetches
#: (``meshy.config.GRPO_TRAINER_FIELDS``): the fetch is an AND-filter.
GRPO_FIELDS = tuple(GRPO_TRAINER_FIELDS)

#: The same columns minus ``advantage``, written when a critic Service owns the
#: advantage instead. Because the trainer's fetch is an AND-filter, leaving the
#: column unwritten is exactly what holds each row back until the critic fills
#: it in -- no extra column and no schema change on either side. See
#: :class:`meshy.worker.critic.CriticWorker`.
GRPO_FIELDS_NO_ADVANTAGE = tuple(f for f in GRPO_FIELDS if f != "advantage")

#: Added when reward shaping is configured: ``reward`` then carries the shaped
#: ``R`` and this carries the unshaped task reward (:mod:`meshy.reward`).
RAW_REWARD_FIELD = "raw_reward"


def resolve_callable(value: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(value):
        return value
    if not isinstance(value, str) or ":" not in value:
        raise TypeError("callable config must be a callable or 'module:attribute' path")
    module_name, attr_path = value.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if not callable(obj):
        raise TypeError(f"configured object {value!r} is not callable")
    return obj


#: What a dataset must expose for a run to be resumable. ``RolloutWorker``
#: rebuilds the dataset per epoch from the same seed, so the permutation is
#: fixed and a resume only needs to skip forward in it -- but the skip has to
#: go through the adapter, because how far "prompt N" is into the underlying
#: records is the adapter's business. ``seek(index)`` is the explicit hook; a
#: writable integer ``index`` (what every built-in adapter uses) is accepted
#: as the implicit one.
_SEEK_HINT = (
    "a resumable dataset must define seek(index) or expose a writable integer "
    "``index`` attribute counting prompts consumed (see "
    "docs/how_to_build_a_recipe.md)"
)


def _require_seekable(dataset: Any) -> None:
    """Fail at the first epoch, not at the first resume, on a fixed dataset."""
    if callable(getattr(dataset, "seek", None)) or hasattr(dataset, "index"):
        return
    raise TypeError(
        f"checkpointing is enabled but {type(dataset).__name__} cannot resume: "
        f"{_SEEK_HINT}"
    )


def _seek_dataset(dataset: Any, index: int) -> None:
    """Advance ``dataset`` to the prompt cursor a checkpoint recorded."""
    seek = getattr(dataset, "seek", None)
    if callable(seek):
        seek(int(index))
        return
    if hasattr(dataset, "index"):
        dataset.index = int(index)
        return
    raise TypeError(f"cannot resume {type(dataset).__name__}: {_SEEK_HINT}")


def grpo_advantage(samples: list[Any]) -> None:
    import torch

    if len(samples) == 1:
        samples[0].advantage = samples[0].reward
        return
    rewards = torch.tensor([sample.reward for sample in samples], dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-8)
    for sample, advantage in zip(samples, advantages):
        sample.advantage = float(advantage.item())


def is_zero_variance_group(samples: list[Any], *, tolerance: float = 1e-8) -> bool:
    """Return whether all raw rewards in a completed group are equal.

    Deliberately the *raw* reward, not the shaped one: the question dynamic
    sampling asks is whether the group's outcomes were informative about the
    task. Length shaping perturbs every sample a little, so keying on the
    shaped reward would report an all-correct group as having variance and
    defeat the filter.
    """
    if not samples:
        return True
    rewards = [
        float(sample.reward if getattr(sample, "raw_reward", None) is None else sample.raw_reward)
        for sample in samples
    ]
    return max(rewards) - min(rewards) <= tolerance


def sample_to_tensordict(sample: Any, weight_version: int):
    import torch
    from tensordict import TensorDict

    raw_reward = getattr(sample, "raw_reward", None)
    return TensorDict(
        {
            "tokens": torch.tensor(list(sample.tokens), dtype=torch.long),
            "logprobs": torch.tensor(list(sample.logprobs), dtype=torch.float32),
            "mask_assistant": torch.tensor(list(sample.masks), dtype=torch.float32),
            # ``None`` when a critic Service owns the advantage: the column is
            # excluded from the write (GRPO_FIELDS_NO_ADVANTAGE), so this
            # placeholder never reaches TQ.
            "advantage": torch.tensor(
                0.0 if sample.advantage is None else float(sample.advantage),
                dtype=torch.float32,
            ),
            "weight_version": torch.tensor(int(weight_version), dtype=torch.int64),
            "reward": torch.tensor(float(sample.reward), dtype=torch.float32),
            "positive_lm_weight": torch.tensor(
                float(getattr(sample, "positive_lm_weight", 0.0)), dtype=torch.float32
            ),
            # Equal to ``reward`` when no shaping is configured; the column is
            # only written when the worker declares it.
            RAW_REWARD_FIELD: torch.tensor(
                float(sample.reward if raw_reward is None else raw_reward),
                dtype=torch.float32,
            ),
            "truncated": torch.tensor(int(bool(getattr(sample, "truncated", False))), dtype=torch.int64),
            "repetition": torch.tensor(int(bool(getattr(sample, "repetition", False))), dtype=torch.int64),
            "mixed_version": torch.tensor(
                int(bool(getattr(sample, "mixed_version", False))), dtype=torch.int64
            ),
        },
        batch_size=[],
    )


def _scalar(value: Any) -> int:
    if getattr(value, "is_nested", False):
        value = value[0]
    return int(value.reshape(-1)[0].item())


class TrajectoryLogger:
    def __init__(self, path: str, *, verbose: bool = False) -> None:
        self.path = os.path.abspath(path)
        self.verbose = bool(verbose)
        self._lock = asyncio.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        logger.info(
            "RolloutWorker: trajectory logging -> {} (verbose={})",
            self.path,
            self.verbose,
        )

    def _record(self, sample: Any, version: int) -> dict[str, Any]:
        record = {
            "timestamp": time.time(),
            # Historical trajectory consumers group samples by training round.
            # The rollout version is the exact source value used by TQ; keep
            # both fields so newer consumers can use the staleness tag directly.
            "round": version + 1,
            "weight_version": version,
            "trajectory": list(sample.messages),
            "ground_truth": sample.ground_truth,
            "reward": sample.reward,
            "advantage": sample.advantage,
            "finish_reason": getattr(sample, "finish_reason", None),
            "truncated": bool(getattr(sample, "truncated", False)),
            "repetition": bool(getattr(sample, "repetition", False)),
            "mixed_version": bool(getattr(sample, "mixed_version", False)),
            "response_tokens": sum(1 for m in sample.masks if m),
        }
        if self.verbose:
            record.update(
                num_tokens=len(list(sample.tokens)),
                tokens=list(sample.tokens),
                logprobs=list(sample.logprobs),
                masks=list(sample.masks),
            )
        return record

    async def write(self, samples: list[Any], version: int) -> None:
        blob = "".join(
            json.dumps(self._record(sample, version), ensure_ascii=False, default=str) + "\n"
            for sample in samples
        )
        async with self._lock:
            await asyncio.to_thread(self._append, blob)

    def _append(self, blob: str) -> None:
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(blob)


class RolloutWorker(TQWorker):
    """Read gen gates, generate grouped trajectories, and append them to TQ."""

    def __init__(
        self,
        *,
        engine: SGLangEngine,
        endpoints_ref: str,
        model_path: str,
        dataset: str | Callable[..., Any],
        dataset_kwargs: dict[str, Any] | None,
        partition_id: str,
        group_size: int,
        train_batch_size: int,
        sampling_params: dict[str, Any] | None,
        reward: str | Callable[[Any], float],
        reward_shaping: str | Callable[..., float] | None = None,
        reward_shaping_kwargs: dict[str, Any] | None = None,
        advantage: str | Callable[[list[Any]], Any] | None = None,
        advantage_kwargs: dict[str, Any] | None = None,
        filter_zero_std_groups: bool = False,
        publish_positive_lm_weights: bool = False,
        oversample_factor: float = 1.0,
        num_epochs: int = 1,
        pacing_window: int | str | None = 1,
        max_running_requests: int = -1,
        poll_interval: float = 2.0,
        trajectory_log: str | None = None,
        verbose_trajectory_log: bool = False,
        external_advantage: bool = False,
        checkpoint: Any = None,
        resume_step: int = 0,
        state_dir: str | None = None,
        service_name: str = "rollout",
        client_factory=None,
        colocation: ColocationManager | None = None,
    ) -> None:
        super().__init__()
        if group_size <= 0 or train_batch_size <= 0:
            raise ValueError("group_size and train_batch_size must be positive")
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if pacing_window == "auto":
            pacing_window = 1
        if pacing_window is not None and int(pacing_window) < 1:
            raise ValueError("pacing_window must be >= 1, 'auto', or None")
        self.engine = engine
        self.colocation = colocation
        self._colocation_request = None
        self.model_path = model_path
        self.dataset_factory = resolve_callable(dataset)
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.group_size = int(group_size)
        self.train_batch_size = int(train_batch_size)
        self.sampling_params = dict(sampling_params or engine.sampling_params)
        self.reward_fn = resolve_callable(reward)
        # Shaping is part of the reward, so it is applied here rather than in
        # the advantage pipeline -- which ``external_advantage`` skips
        # entirely. See :mod:`meshy.reward`.
        self.reward_shaping_fn = (
            resolve_callable(reward_shaping) if reward_shaping is not None else None
        )
        self.reward_shaping_kwargs = dict(reward_shaping_kwargs or {})
        # A critic Service computes the advantage from its own value net, so
        # the rollout must neither compute nor publish the column: writing a
        # group-normalised placeholder would satisfy the trainer's AND-filter
        # and let rows through before the critic ever saw them.
        self.external_advantage = bool(external_advantage)
        if self.external_advantage:
            if advantage is not None:
                raise ValueError(
                    "external_advantage=True hands the advantage to a critic "
                    "Service; an inline advantage callable would be ignored"
                )
            self.advantage_fn = None
            fields = GRPO_FIELDS_NO_ADVANTAGE
        else:
            self.advantage_fn = grpo_advantage if advantage is None else resolve_callable(advantage)
            fields = GRPO_FIELDS
        # Declared only when there is something to distinguish it from
        # ``reward``: the trainer's fetch is an AND-filter, so a column no
        # consumer asked for is harmless but one nobody writes is fatal.
        if self.reward_shaping_fn is not None:
            fields = fields + (RAW_REWARD_FIELD,)
        self.rollout_fields = fields
        self.publish_positive_lm_weights = publish_positive_lm_weights
        if publish_positive_lm_weights:
            self.rollout_fields = fields + ("positive_lm_weight",)
        self.advantage_kwargs = dict(advantage_kwargs or {})
        self.filter_zero_std_groups = bool(filter_zero_std_groups)
        if oversample_factor < 1.0:
            raise ValueError("oversample_factor must be >= 1")
        # Replacement budget for groups dropped by ``filter_zero_std_groups``,
        # in multiples of the dataset. See :meth:`_run_rollouts`.
        self.oversample_factor = float(oversample_factor)
        self.groups_seen = 0
        self.groups_filtered = 0
        self.num_epochs = int(num_epochs)
        self.pacing_window = None if pacing_window is None else int(pacing_window)
        self.max_running_requests = int(max_running_requests)
        self.poll_interval = float(poll_interval)
        self.gates_seen = 0
        self.weight_version = 0
        self.samples_started = 0
        # ── resumable data position ──────────────────────────────────────
        from meshy.config import GlobalCheckpointConfig

        self.checkpoint = GlobalCheckpointConfig.coerce(checkpoint)
        self.resume_step = int(resume_step)
        self.service_name = service_name
        self._state_dir = state_dir
        self._store = None
        #: Where the dataset cursor stands, mirrored onto the Worker so a
        #: snapshot taken from the gate loop can see it. ``_dataset_index``
        #: counts prompts *pulled off the dataset*, not prompts trained on, so
        #: a resume skips rather than replays: the groups still generating at
        #: the checkpoint, plus any tail of the current dataset batch not yet
        #: dispatched. Skipping keeps a resumed run from training twice on the
        #: same prompt, which duplicating would do to every one of them.
        self._epoch = 0
        self._dataset_index = 0
        self._filtered_at_epoch_start = 0
        self._restore()
        self.trajectory_logger = (
            TrajectoryLogger(trajectory_log, verbose=verbose_trajectory_log)
            if trajectory_log
            else None
        )

        gen_gate_fields = ("gate_step", "weight_version")
        gen_gate_partition = "gen_gate"
        gen_gate_task = "gen_gate"

        self.configure_tq(
            endpoints_ref=endpoints_ref,
            input=None,
            controls={
                "gen_gate": TQInput(
                    partition=gen_gate_partition,
                    fields=gen_gate_fields,
                    batch_size=1,
                    consumer=gen_gate_task,
                    clear_after_success=True,
                )
            },
            outputs={
                "rollouts": TQOutput(
                    fields=self.rollout_fields,
                    new_rows=True,
                    partition=partition_id,
                )
            },
            poll_interval=poll_interval,
            client_factory=client_factory,
        )

    def run(self) -> None:
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        await self.open_tq()
        try:
            await self._run_rollouts()
        finally:
            await self.close_tq()

    # ── global checkpoint ────────────────────────────────────────────────
    @property
    def store(self):
        """Lazily-built :class:`~meshy.service.checkpoint.StateStore`."""
        if self._store is None:
            from meshy.service.checkpoint import StateStore

            if not self._state_dir:
                raise RuntimeError(
                    "RolloutWorker checkpointing needs a state_dir; the Service "
                    "did not wire one"
                )
            self._store = StateStore(self._state_dir)
        return self._store

    def _restore(self) -> None:
        """Re-seed the data position and the pacing counters from a shard.

        ``gates_seen`` is deliberately restored to ``resume_step`` rather than
        the ``resume_step + 1`` the snapshot recorded. TransferQueue is
        RAM-only, so a resumed run's ``gen_gate`` partition starts empty and
        the trainer re-publishes a genesis pulse *at* ``resume_step`` to
        re-establish the stream. Setting the counter one lower is what makes
        this worker expect exactly that pulse, keeping
        :meth:`try_advance_gate`'s ordering check satisfied and re-granting
        the last window's sample budget exactly once.
        """
        from meshy.service.checkpoint import log_resume

        if self.resume_step <= 0:
            return
        payload = self.store.payload(self.service_name, self.resume_step)
        if not payload:
            raise FileNotFoundError(
                f"resuming at global step {self.resume_step} but "
                f"{self.store.meta_path(self.service_name, self.resume_step)} "
                "is missing; the shard is incomplete"
            )
        self._epoch = int(payload["epoch"])
        self._dataset_index = int(payload["dataset_index"])
        self._filtered_at_epoch_start = int(payload["filtered_at_epoch_start"])
        self.groups_seen = int(payload["groups_seen"])
        self.groups_filtered = int(payload["groups_filtered"])
        self.samples_started = int(payload["samples_started"])
        self.weight_version = int(payload["weight_version"])
        self.gates_seen = self.resume_step
        log_resume(
            self.service_name,
            self.resume_step,
            f"epoch={self._epoch} dataset_index={self._dataset_index} "
            f"samples_started={self.samples_started} v{self.weight_version}",
        )

    def _snapshot(self, global_step: int) -> None:
        """Commit the rollout's shard for ``global_step``.

        Called from the gate loop, which is the rollout's only point of
        contact with the run's clock -- and the right one: gate ``N`` is
        published by the trainer *after* its step ``N`` is durable, so a
        rollout shard stamped ``N`` can never describe a window the trainer
        has not finished.
        """
        self.store.write_meta(
            self.service_name,
            global_step,
            {
                "epoch": self._epoch,
                "dataset_index": self._dataset_index,
                "filtered_at_epoch_start": self._filtered_at_epoch_start,
                "groups_seen": self.groups_seen,
                "groups_filtered": self.groups_filtered,
                "gates_seen": self.gates_seen,
                "samples_started": self.samples_started,
                "weight_version": self.weight_version,
            },
        )
        self.store.mark_complete(self.service_name, global_step)
        self.store.prune(self.service_name, self.checkpoint.keep_latest_k)
        logger.info(
            "RolloutWorker checkpointed global step {} (epoch {}, prompt {})",
            global_step,
            self._epoch,
            self._dataset_index,
        )

    def _generation_budget(self) -> int:
        if self.gates_seen == 0:
            return 0
        assert self.pacing_window is not None
        return (self.gates_seen - 1 + self.pacing_window) * self.train_batch_size

    async def try_advance_gate(self) -> bool:
        data = await self.read_tq_control("gen_gate")
        if data is None:
            return False
        step = _scalar(data["gate_step"])
        version = _scalar(data["weight_version"])
        expected = self.gates_seen
        if step != expected:
            logger.warning(
                "gen_gate stream out of order: expected {}, got {} (weight v{})",
                expected,
                step,
                version,
            )
        previous = self.weight_version
        self.gates_seen += 1
        self.weight_version = version
        logger.info(
            "RolloutWorker: gate {} up, version {} -> {} ({} samples started)",
            step,
            previous,
            version,
            self.samples_started,
        )
        # The gate step *is* the run's global window index, so this is where
        # the rollout learns a checkpoint boundary has passed. Re-consuming
        # the resume pulse must not re-snapshot on top of the shard it was
        # restored from, hence the ``>`` rather than a bare ``due()``.
        if self.checkpoint.due(step) and step > self.resume_step:
            self._snapshot(step)
        return True

    async def drain_gates(self) -> None:
        while await self.try_advance_gate():
            pass

    async def acquire_generation_slot(self, num_samples: int) -> int:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        # The initial gate establishes the version that inference is allowed to
        # use.  An ungated pacing policy may run ahead only after that gate.
        while self.gates_seen == 0:
            if self.stopped:
                raise asyncio.CancelledError
            if not await self.try_advance_gate():
                await asyncio.sleep(min(self.poll_interval, 2.0))
        if self.pacing_window is None:
            await self.drain_gates()
            self.samples_started += num_samples
            return self.weight_version
        while self.samples_started + num_samples > self._generation_budget():
            if self.stopped:
                raise asyncio.CancelledError
            if not await self.try_advance_gate():
                await asyncio.sleep(min(self.poll_interval, 2.0))
        self.samples_started += num_samples
        return self.weight_version

    async def _ensure_colocation(self) -> None:
        """Ensure the fallback rollout role currently owns the GPU token."""
        if self.colocation is None or self.colocation.owns_gpu:
            return
        if self._colocation_request is None:
            request_id = f"rollout:fallback:{self.gates_seen}"
            self._colocation_request = self.colocation.request_gpu(
                request_id=request_id, kind=RequestKind.FALLBACK
            )
        await asyncio.to_thread(self.colocation.wait_for_grant, self._colocation_request)

    async def rollout_group(self, builder: Any, prompt: Any, weight_version: int) -> list[Any] | None:
        await self._ensure_colocation()
        group = [copy.deepcopy(prompt) for _ in range(self.group_size)]

        async def rollout_one(sample: Any) -> Any:
            generation = await self.engine.generate(
                list(sample.tokens),
                sampling_params=self.sampling_params,
            )
            tokens, logprobs = generation
            builder.append_tokens(sample, "assistant", tokens, logprobs)
            self._stamp_sample(sample, generation)
            reward = float(self.reward_fn(sample))
            if self.reward_shaping_fn is not None:
                # ``reward`` becomes the shaped R the critic and the GAE
                # recursion consume; the unshaped value rides along so the
                # solve-rate metrics stay about correctness.
                sample.raw_reward = reward
                sample.reward = float(
                    self.reward_shaping_fn(reward, sample, **self.reward_shaping_kwargs)
                )
            else:
                sample.reward = reward
            return sample

        group = await asyncio.gather(*(rollout_one(sample) for sample in group))
        if getattr(self, "publish_positive_lm_weights", False):
            from meshy.utils.positive_lm import assign_positive_lm_weights
            assign_positive_lm_weights(group)
        self.groups_seen += 1
        if self.filter_zero_std_groups:
            rewards = [float(sample.reward) for sample in group]
            if is_zero_variance_group(group):
                self.groups_filtered += 1
                logger.debug(
                    "RolloutWorker: dropping zero-variance group {} (reward={})",
                    self.groups_seen,
                    rewards[0] if rewards else None,
                )
                return None
        if self.advantage_fn is not None:
            advantages = self.advantage_fn(group, **self.advantage_kwargs)
            if advantages is not None:
                values = list(advantages)
                if len(values) != len(group):
                    raise ValueError("advantage function result length must match rollout group")
                for sample, value in zip(group, values):
                    sample.advantage = float(value)
        return group

    @staticmethod
    def _stamp_sample(sample: Any, generation: Any) -> None:
        """Record the per-sample quality flags the trainer turns into metrics.

        ``generation`` is normally an :class:`~meshy.engine.sglang.Generation`;
        a bare ``(tokens, logprobs)`` tuple (legacy engines, test doubles)
        leaves the flags at their defaults.
        """
        finish_reason = getattr(generation, "finish_reason", None)
        sample.finish_reason = finish_reason
        # SGLang reports ``length`` when the response hit ``max_new_tokens``.
        sample.truncated = bool(
            getattr(generation, "truncated", finish_reason == "length")
        )
        # Every abort/resume cycle happened because a colocated training step
        # took the card; the resumed tokens come from the *next* weights.
        sample.mixed_version = int(getattr(generation, "continuations", 0) or 0) > 0
        response = sample.messages[-1]["content"] if sample.messages else ""
        sample.repetition = has_repetition(response)

    async def _write_rollout_group(self, group: list[Any], version: int) -> None:
        from meshy.transferqueue import adapter

        rows = [sample_to_tensordict(sample, version) for sample in group]
        data = adapter.samples_to_td(rows, self.rollout_fields)
        if self.trajectory_logger is not None:
            await self.trajectory_logger.write(group, version)
        await self.write_tq_output("rollouts", data)

    def _oversample_budget(self, dataset_prompts: int | None) -> int | None:
        """Extra prompts this epoch may spend replacing filtered groups.

        ``filter_zero_std_groups`` drops a group after it has been generated,
        so without replacement an epoch that filters heavily yields fewer
        windows than it has prompts for. Replacement here is *implicit* rather
        than a refill buffer: the rollout does not drain a window before
        starting the next (``pacing_window=None``), and every consumer takes a
        fixed row count off TQ, so a dropped group is simply made up by the
        prompts already in flight behind it. All this budget does is bound how
        far the epoch may over-run its nominal length -- which is what stops a
        stretch of uniformly-solved prompts from burning the dataset.

        ``None`` when no bound applies: the filter is off, no replacement was
        asked for, or the dataset cannot report its own size (in which case
        there is nothing to take a fraction of, and an unbounded epoch is the
        pre-existing behaviour rather than a new risk).
        """
        if not self.filter_zero_std_groups or self.oversample_factor <= 1.0:
            return None
        if not dataset_prompts:
            logger.warning(
                "RolloutWorker: oversample_factor={} ignored -- the dataset does "
                "not expose ``n_prompts``",
                self.oversample_factor,
            )
            return None
        return int(dataset_prompts * (self.oversample_factor - 1.0))

    async def _run_rollouts(self) -> None:
        from meshy.utils.sample import SampleBuilder

        builder = SampleBuilder(self.model_path)
        max_groups = (
            max(1, math.ceil(self.max_running_requests / self.group_size))
            if self.max_running_requests > 0
            else None
        )
        semaphore = asyncio.Semaphore(max_groups) if max_groups is not None else None
        tasks: set[asyncio.Task[Any]] = set()

        async def execute(prompt: Any, version: int) -> None:
            try:
                group = await self.rollout_group(builder, prompt, version)
                if group is not None:
                    await self._write_rollout_group(group, version)
            except Exception as exc:
                logger.exception("RolloutWorker group failed: {}", exc)
            finally:
                if semaphore is not None:
                    semaphore.release()

        # A resumed run re-enters the epoch it stopped in and skips forward to
        # the prompt it had reached. It must *not* reshuffle with an
        # epoch-derived seed: the dataset is rebuilt from the same seed each
        # epoch, so the permutation is fixed, and seeking into it is the only
        # way a resumed run stays on the original data order.
        resuming = self.resume_step > 0
        for epoch in range(self._epoch, self.num_epochs):
            if self.stopped:
                break
            self._epoch = epoch
            dataset = self.dataset_factory(**self.dataset_kwargs)
            if self.checkpoint.enabled:
                _require_seekable(dataset)
            if resuming:
                _seek_dataset(dataset, self._dataset_index)
                logger.info(
                    "RolloutWorker: epoch {} fast-forwarded to prompt {}; the "
                    "groups in flight at the checkpoint are skipped, not replayed",
                    epoch,
                    self._dataset_index,
                )
            else:
                self._dataset_index = 0
                self._filtered_at_epoch_start = self.groups_filtered
            resuming = False
            budget = self._oversample_budget(getattr(dataset, "n_prompts", None))
            filtered_at_epoch_start = self._filtered_at_epoch_start
            while not self.stopped:
                prompts = dataset.next_batch(builder)
                self._dataset_index = int(
                    getattr(dataset, "index", self._dataset_index + len(prompts))
                )
                if not prompts:
                    logger.info(
                        "RolloutWorker dataset exhausted (epoch {}); {} of {} groups "
                        "filtered as zero-variance",
                        epoch,
                        self.groups_filtered,
                        self.groups_seen,
                    )
                    break
                if budget is not None:
                    spent = self.groups_filtered - filtered_at_epoch_start
                    if spent > budget:
                        logger.info(
                            "RolloutWorker: oversample budget exhausted (epoch {}): "
                            "{} groups filtered, budget {}",
                            epoch,
                            spent,
                            budget,
                        )
                        break
                for prompt in prompts:
                    version = await self.acquire_generation_slot(self.group_size)
                    if semaphore is not None:
                        await semaphore.acquire()
                    task = asyncio.create_task(execute(prompt, version))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tasks)


__all__ = [
    "GRPO_FIELDS",
    "GRPO_FIELDS_NO_ADVANTAGE",
    "RAW_REWARD_FIELD",
    "RolloutWorker",
    "grpo_advantage",
    "is_zero_variance_group",
    "resolve_callable",
    "sample_to_tensordict",
]
