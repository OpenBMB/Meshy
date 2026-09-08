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
    """Return whether all raw rewards in a completed group are equal."""
    if not samples:
        return True
    rewards = [float(sample.reward) for sample in samples]
    return max(rewards) - min(rewards) <= tolerance


def sample_to_tensordict(sample: Any, weight_version: int):
    import torch
    from tensordict import TensorDict

    return TensorDict(
        {
            "tokens": torch.tensor(list(sample.tokens), dtype=torch.long),
            "logprobs": torch.tensor(list(sample.logprobs), dtype=torch.float32),
            "mask_assistant": torch.tensor(list(sample.masks), dtype=torch.float32),
            "advantage": torch.tensor(float(sample.advantage), dtype=torch.float32),
            "weight_version": torch.tensor(int(weight_version), dtype=torch.int64),
            "reward": torch.tensor(float(sample.reward), dtype=torch.float32),
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
        advantage: str | Callable[[list[Any]], Any] | None = None,
        advantage_kwargs: dict[str, Any] | None = None,
        filter_zero_std_groups: bool = False,
        num_epochs: int = 1,
        pacing_window: int | str | None = 1,
        max_running_requests: int = -1,
        poll_interval: float = 2.0,
        trajectory_log: str | None = None,
        verbose_trajectory_log: bool = False,
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
        self.advantage_fn = grpo_advantage if advantage is None else resolve_callable(advantage)
        self.advantage_kwargs = dict(advantage_kwargs or {})
        self.filter_zero_std_groups = bool(filter_zero_std_groups)
        self.groups_seen = 0
        self.groups_filtered = 0
        self.num_epochs = int(num_epochs)
        self.pacing_window = None if pacing_window is None else int(pacing_window)
        self.max_running_requests = int(max_running_requests)
        self.poll_interval = float(poll_interval)
        self.gates_seen = 0
        self.weight_version = 0
        self.samples_started = 0
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
                    fields=GRPO_FIELDS,
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
            sample.reward = self.reward_fn(sample)
            return sample

        group = await asyncio.gather(*(rollout_one(sample) for sample in group))
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
        data = adapter.samples_to_td(rows, GRPO_FIELDS)
        if self.trajectory_logger is not None:
            await self.trajectory_logger.write(group, version)
        await self.write_tq_output("rollouts", data)

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

        for epoch in range(self.num_epochs):
            if self.stopped:
                break
            dataset = self.dataset_factory(**self.dataset_kwargs)
            while not self.stopped:
                prompts = dataset.next_batch(builder)
                if not prompts:
                    logger.info("RolloutWorker dataset exhausted (epoch {})", epoch)
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
    "RolloutWorker",
    "grpo_advantage",
    "is_zero_variance_group",
    "resolve_callable",
    "sample_to_tensordict",
]
