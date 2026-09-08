"""Meshy Titan metrics adapter.

The numerical training implementation still comes from the compatibility
backend.  This module is deliberately an adapter: it observes the sample list
the trainer receives (plus the per-sample quality stamps the rollout worker
ships through TransferQueue: ``reward``, ``truncated``, ``repetition``,
``mixed_version``) and the scalar metrics returned by the loss, then publishes
derived metrics to TensorBoard.

Tag hygiene: a TensorBoard tag must be *either* a scalar *or* a histogram.
Writing both under one name makes the scalars plugin fail with HTTP 500 for
that tag (the histogram tensor cannot be read as a scalar), so histograms
live under the ``hist/`` prefix.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch

from meshy.backend.titan import build_forge_config
from meshy.backend.titan.trainer import TitanTrainer as _BackendTitanTrainer

#: Histograms are written under this prefix so they never share a tag with a
#: scalar (see module docstring).
HISTOGRAM_PREFIX = "hist/"
#: Per-step histogram sample cap: the full per-token log-prob list of a
#: 128k-context batch is tens of millions of floats.
HISTOGRAM_MAX_VALUES = 200_000


class _TensorBoard:
    """Small lazy writer wrapper; unavailable TensorBoard must not stop training."""

    def __init__(self, log_dir: str | None, enabled: bool) -> None:
        self.writer = None
        if not enabled or not log_dir:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        except (ImportError, ModuleNotFoundError):
            self.writer = None

    def log(self, metrics: dict[str, float], step: int, histograms: dict[str, list[float]] | None = None) -> None:
        if self.writer is None:
            return
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self.writer.add_scalar(name, float(value), global_step=step)
        for name, values in (histograms or {}).items():
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float32)
            tensor = tensor[torch.isfinite(tensor)]
            if tensor.numel() == 0:
                continue
            if tensor.numel() > HISTOGRAM_MAX_VALUES:
                stride = math.ceil(tensor.numel() / HISTOGRAM_MAX_VALUES)
                tensor = tensor[::stride]
            if not name.startswith(HISTOGRAM_PREFIX):
                name = HISTOGRAM_PREFIX + name
            self.writer.add_histogram(name, tensor, global_step=step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _sample_field(sample: Any, field: str, default: Any = None) -> Any:
    if hasattr(sample, "get"):
        return sample.get(field, default)
    return getattr(sample, field, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    try:
        return list(value)
    except TypeError:
        return [value]


def _values(samples: list[Any], field: str) -> list[float]:
    out: list[float] = []
    for sample in samples:
        value = _sample_field(sample, field, None)
        if value is None:
            continue
        for x in _as_list(value):
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                pass
    return out


def _flags(samples: list[Any], field: str) -> list[float] | None:
    """Per-sample 0/1 stamps; ``None`` when the batch does not carry the column."""
    values = _values(samples, field)
    if len(values) != len(samples):
        return None
    return [1.0 if v > 0.5 else 0.0 for v in values]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class TitanTrainer(_BackendTitanTrainer):
    """TitanTrainer with Meshy metrics and TensorBoard publication."""

    def __init__(self, *args: Any, tensorboard_log_dir: str | None = None,
                 tensorboard_enabled: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Only rank zero writes an event stream.  In non-distributed unit tests
        # dist is uninitialized and the local process is treated as rank zero.
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except RuntimeError:
            pass
        self._tensorboard = _TensorBoard(
            tensorboard_log_dir if rank == 0 else None,
            tensorboard_enabled,
        )

    def restore_to_gpu(self) -> None:
        # Decoder.freqs_cis is a non-persistent root buffer.  The legacy
        # model move restores parameters but can leave this cache on CPU after
        # an offload; move it explicitly before the next forward pass.
        super().restore_to_gpu()
        for model_part in getattr(self, "model_parts", ()):
            # FSDP wrappers expose the decoder root one level below the part;
            # walk all modules so the non-persistent cache is found in either
            # layout.
            modules = model_part.modules() if hasattr(model_part, "modules") else (model_part,)
            for module in modules:
                cache = getattr(module, "freqs_cis", None)
                if torch.is_tensor(cache) and cache.device != self.device:
                    module.freqs_cis = cache.to(self.device)
                rope = getattr(module, "rope", None)
                rope_cache = getattr(rope, "cache", None) if rope is not None else None
                if torch.is_tensor(rope_cache) and rope_cache.device != self.device:
                    rope.cache = rope_cache.to(self.device)

    @staticmethod
    def _rollout_metrics(
        samples: list[Any], seq_len: int, current_version: int | None = None
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Batch-level rollout statistics from the (global) sample list.

        ``current_version`` is the weight version the trainer is about to
        update (its step counter); staleness is measured against it.
        """
        rewards = _values(samples, "reward")
        advantages = _values(samples, "advantage")
        weight_versions = _values(samples, "weight_version")
        lengths: list[float] = []
        response_lengths: list[float] = []
        logprobs: list[float] = []
        for sample in samples:
            token_list = _as_list(_sample_field(sample, "tokens", None))
            if token_list:
                lengths.append(float(len(token_list)))
            raw_mask = _sample_field(sample, "mask_assistant", None)
            if raw_mask is None:
                raw_mask = _sample_field(sample, "masks", None)
            mask_values = [float(v) > 0 for v in _as_list(raw_mask)]
            if raw_mask is not None:
                response_lengths.append(float(sum(mask_values)))
            lp = _as_list(_sample_field(sample, "logprobs", None))
            if lp:
                # Only the sampled (assistant) tokens carry behaviour-policy
                # log-probs; prompt positions are 0.0 placeholders.
                if len(mask_values) == len(lp):
                    logprobs.extend(float(v) for v, keep in zip(lp, mask_values) if keep)
                else:
                    logprobs.extend(float(v) for v in lp)

        m: dict[str, float] = {}
        if response_lengths:
            m.update({
                "rollout/response_len/max": max(response_lengths),
                "rollout/response_len/mean": _mean(response_lengths),
                "rollout/response_len/median": _percentile(response_lengths, 0.5),
                "rollout/response_len/min": min(response_lengths),
                "rollout/response_len/p0": _percentile(response_lengths, 0.0),
                "rollout/response_len/p25": _percentile(response_lengths, 0.25),
                "rollout/response_len/p50": _percentile(response_lengths, 0.5),
                "rollout/response_len/p75": _percentile(response_lengths, 0.75),
                "rollout/response_len/p90": _percentile(response_lengths, 0.9),
                "rollout/response_len/p95": _percentile(response_lengths, 0.95),
                "rollout/response_len/p99": _percentile(response_lengths, 0.99),
                "rollout/response_lengths": _mean(response_lengths),
            })
        if lengths:
            m["rollout/total_lengths"] = _mean(lengths)
            # Samples the *trainer* will cut at ``seq_len`` (distinct from the
            # generation-side truncation below).
            m["rollout/seq_len_truncated_ratio"] = sum(v >= seq_len for v in lengths) / len(lengths)

        # Generation-side stamps from the rollout worker.
        truncated = _flags(samples, "truncated")
        if truncated is not None:
            # finish_reason == "length": the response hit max_new_tokens.
            m["rollout/truncated_ratio"] = _mean(truncated)
            m["rollout/truncated"] = m["rollout/truncated_ratio"]
        repetition = _flags(samples, "repetition")
        if repetition is not None:
            # Fraction of samples whose response tail is degenerate repetition
            # (zlib compression ratio of the last 10k chars > 10, as in Miles).
            m["rollout/repetition_frac"] = _mean(repetition)

        if advantages:
            nonzero_adv = [v for v in advantages if abs(v) > 1e-8]
            m.update({
                "grpo_metrics/effective_count": float(len(nonzero_adv)),
                "grpo_metrics/effective_ratio": len(nonzero_adv) / len(advantages),
                "grpo_metrics/zero_advantage_ratio": 1.0 - len(nonzero_adv) / len(advantages),
                "rollout/advantages": _mean(advantages),
                "rollout/advantages_abs_mean": _mean([abs(v) for v in advantages]),
            })
        if rewards:
            solved = [v > 0.5 for v in rewards]
            m.update({
                "grpo_metrics/solve_all": float(all(solved)),
                "grpo_metrics/solve_none": float(not any(solved)),
                "rollout/raw_reward_mean": _mean(rewards),
                "rollout/raw_reward": _mean(rewards),
                "rollout/rewards": _mean(rewards),
                "rollout/reward_max": max(rewards),
                "rollout/reward_min": min(rewards),
                "rollout/pass_rate": _mean([1.0 if s else 0.0 for s in solved]),
            })
        if logprobs:
            m.update({
                "rollout/log_probs": _mean(logprobs),
                "rollout/rollout_log_probs": _mean(logprobs),
            })
        if weight_versions:
            newest = max(weight_versions)
            m.update({
                "rollout/weight_version/mean": _mean(weight_versions),
                "rollout/weight_version/min": min(weight_versions),
                "rollout/weight_version/max": newest,
                "rollout/weight_version/median": _percentile(weight_versions, 0.5),
                # Share of samples generated against an older version than the
                # newest one in this batch (was a batch-level 0/1 indicator).
                "rollout/weight_version/stale_ratio": _mean(
                    [1.0 if v < newest else 0.0 for v in weight_versions]
                ),
                "rollout/weight_version/num_versions": float(len(set(weight_versions))),
            })
            if current_version is not None:
                staleness = [float(current_version) - v for v in weight_versions]
                m["rollout/weight_version/staleness_mean"] = _mean(staleness)
                m["rollout/weight_version/staleness_max"] = max(staleness)
        mixed = _flags(samples, "mixed_version")
        if mixed is not None:
            # Fraction of *samples* whose generation was paused for a training
            # step and resumed on the next weights (Miles semantics).
            m["rollout/weight_version/mixed_version_ratio"] = _mean(mixed)

        hist = {
            "rollout/response_lengths": response_lengths,
            "rollout/total_lengths": lengths,
        }
        if rewards:
            hist["rollout/rewards"] = rewards
        if advantages:
            hist["rollout/advantages"] = advantages
        if logprobs:
            hist["rollout/log_probs"] = logprobs
        return m, hist

    def _training_metrics(self, result: dict[str, float]) -> dict[str, float]:
        """Map metrics produced by the loss implementation to public tags."""
        out: dict[str, float] = {}
        mappings = {
            "pg_loss": ("train/pg_loss", "train/loss"),
            # ``ratio_mean`` is E[new/old] over loss tokens; it is ~1 by
            # construction under the behaviour policy, so read ``ppo_kl`` /
            # ``log_ratio_abs_mean`` for how far off-policy the batch is.
            "ratio_mean": ("train/ois", "train/tis"),
            "clip_frac": "train/pg_clipfrac",
            "tis_masked_frac": ("train/tis_masked_token_ratio", "train/tis_clipfrac"),
            "ppo_kl": "train/ppo_kl",
            "log_ratio_abs_mean": "train/log_ratio_abs_mean",
            "train_rollout_kl": "train/train_rollout_kl",
            "train_rollout_logdiff_abs": "train/train_rollout_logprob_abs_diff",
            "ess_ratio": "train/ess_ratio",
            "entropy": ("train/entropy", "rollout/entropy"),
            "grad_norm": "train/grad_norm",
            "grad_norm_max": "train/grad_norm_max",
            "grad_norm_last": "train/grad_norm_last",
            "num_mini_batches": "train/num_mini_batches",
        }
        for source, target in mappings.items():
            if source in result:
                targets = target if isinstance(target, tuple) else (target,)
                for name in targets:
                    out[name] = float(result[source])
        # These are only emitted when an optimizer actually exposes its LR.
        lr_values = []
        for optimizer in getattr(self, "optimizers", []):
            for group in getattr(optimizer, "param_groups", []):
                if "lr" in group:
                    lr_values.append(float(group["lr"]))
        for index, value in enumerate(lr_values[:2]):
            out[f"train/lr-pg_{index}"] = value
        if lr_values:
            out["train/lr"] = lr_values[0]
        return out

    def train_step(
        self,
        samples: list[Any],
        *,
        plan: Any = None,
        step_schedule: bool = True,
        rollout_samples: list[Any] | None = None,
    ) -> dict[str, float]:
        """One training step plus TensorBoard publication.

        ``rollout_samples`` is the *global* batch (before the DP split) when
        the caller has it; the rollout statistics are computed over it so a
        multi-rank run reports the whole batch rather than rank zero's slice.
        """
        started = time.perf_counter()
        rollout, hist = self._rollout_metrics(
            rollout_samples if rollout_samples is not None else samples,
            self.seq_len,
            current_version=int(getattr(self, "step", 0)),
        )
        result = dict(super().train_step(samples, plan=plan, step_schedule=step_schedule))
        metrics = dict(result)
        metrics.update(rollout)
        metrics.update(self._training_metrics(result))
        total_time = max(time.perf_counter() - started, 1e-9)
        tokens = rollout.get("rollout/response_lengths")
        perf: dict[str, float] = {"perf/step_time": total_time}
        if tokens is not None:
            perf.update({
                "perf/tokens_per_gpu_per_sec": float(tokens) / max(total_time, 1e-9),
                "perf/effective_tokens_per_gpu_per_sec": float(tokens) / max(total_time, 1e-9),
            })
        longest = rollout.get("rollout/response_len/max")
        if longest is not None:
            perf["perf/longest_sample_tokens_per_sec"] = float(longest) / max(total_time, 1e-9)
        pack = result.get("time/train/pack_batch")
        old_lp = result.get("time/train/old_logprobs")
        timed_values = [float(v) for k, v in result.items() if k.startswith("time/train/")]
        train_time = sum(timed_values)
        if pack is not None:
            perf["perf/data_preprocess_time"] = float(pack)
        if old_lp is not None:
            perf["perf/log_probs_time"] = float(old_lp)
        if timed_values:
            perf.update({"perf/train_time": train_time, "perf/actor_train_time": train_time})
            if tokens is not None:
                perf["perf/actor_train_tok_per_s"] = float(tokens) / max(train_time, 1e-9)
                perf["perf/actor_train_tok_per_s_8gpu"] = perf["perf/actor_train_tok_per_s"] * 8.0
                if old_lp is not None and float(getattr(self, "model_param_count", 0)) > 0:
                    perf["perf/actor_train_tflops"] = 6.0 * float(self.model_param_count) * float(tokens) / max(train_time, 1e-9) / 1e12
                    perf["perf/actor_train_tflops_8gpu"] = perf["perf/actor_train_tflops"] * 8.0
                    perf["perf/log_probs_tflops"] = 2.0 * float(self.model_param_count) * float(tokens) / max(float(old_lp), 1e-9) / 1e12
            perf["perf/wait_time_ratio"] = max(0.0, total_time - train_time) / total_time
        metrics.update(perf)
        step = int(metrics.get("step", getattr(self, "step", 0)))
        self._tensorboard.log(metrics, step, hist)
        return metrics

    def close(self) -> None:
        self._tensorboard.close()


__all__ = ["HISTOGRAM_PREFIX", "TitanTrainer", "build_forge_config"]
