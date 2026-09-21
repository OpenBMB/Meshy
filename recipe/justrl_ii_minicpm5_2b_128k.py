"""PPO + independent critic (VAPO-GAE) for MiniCPM5-2B at 128k, one 8-card node.

The GRPO recipes derive the advantage from group-normalised reward, so a group
whose samples all pass (or all fail) carries no gradient. This recipe replaces
that with a separate PPO critic: the advantage comes from a value baseline, so
zero-variance groups still carry signal (``justrl_ii_recipe.md`` §1).

Recipe conformance (``justrl_ii_recipe.md``):

| §3 item | value | here |
|---|---|---|
| base model | MiniCPM5-2B | `openbmb/MiniCPM5-2B-SFT` |
| dataset | category-aligned S9, 19.6k | `S9_DATASET_PATH` |
| batch | 60 prompts × 8 = GBS 480 | `ROLLOUT_BATCH` × `GROUP_SIZE` |
| seq_len / response cap | 131072 / 126976 | same |
| parallelism | TP1 · CP4 | same, per replica |
| lr | 1e-6 constant, adam eps 1e-8 | same |
| sampling | T=1.0, top_p=1.0 | same |
| clip-higher | 0.2 / 0.28 | same |
| TIS | on | `use_tis=True` |
| KL | none (kl_coef=0) | no reference model at all |
| critic cold start | ~30 windows, actor frozen | `COLD_START_WINDOWS` |
| nodes | actor 2 · critic 2 (32×H800) | 8×H800, serial colocate |
| overlong soft penalty | buffer 25395, f=1.0 | `reward_shaping` |
| dynamic sampling + 2× oversample | on | `filter_zero_std_groups` (off, see there) |
| positive_lm_loss | 0.1 | **absent** |
| in-loop eval | every 5 steps, AIME ×3 ×4 | **absent -- read this first** |
| ckpt | every 10 steps, rotated | `CHECKPOINT`, see §"Disk" |
| partial rollout | on | on, via SGLang abort/continue |

Where each piece of the VAPO advantage runs
-------------------------------------------

The three stages own three disjoint parts of it, and each hands the next
exactly one column:

    rollout   r = MATH reward; R = r + overlong penalty
              -> writes `reward` = R, `raw_reward` = r
    critic    V = value net over (prompt + response), scored with the
              *pre-update* weights
              -> writes `values`; then fits V to [R, R, ..., R]
    trainer   delta[t] = R[t] + gamma*V[t+1] - V[t]
              A[t]     = delta[t] + gamma*lambda*A[t+1],
              lambda   = 1 - 1/(alpha*L)
              -> per-token A straight into the PPO ratio (`enable_gae`)

Everything above lives on one index grid: position `t` is the state that has
consumed tokens `<= t` and is about to emit token `t+1`. That is the grid
`new_lp` is defined on, so `A[t]` baselines the action whose probability the
ratio measures. The critic trains its value head on the same grid
(`critic/data.py::make_batch(shift=True)`); a value net fitted one token out
of step would be a plausible-looking baseline for the wrong action.

Scaled to one node with serial colocate: all 8 GPUs are shared among inference,
critic, and actor_train in a serial pipeline. SGLang runs in FALLBACK mode and
releases its KV-cache memory when the critic or trainer needs the node; the three
stages execute in sequence (rollout → critic → train) rather than overlapping.
This trades throughput (wall-clock per step = rollout + critic + train rather than
max of the three) for resource efficiency — useful when only one node is available.
CP4 × FSDP2 (`cp_degree=4, dp_shard_degree=2`) covers all 8 cards for both
actor and critic.

Run tempo
---------

One window = `BATCH_SIZE` = 480 rows = 60 prompts × 8 samples. The trainer's
`batch_size`, the critic's `score_batch_size` and the rollout's per-gate budget
are all that same number, so the three stages advance in lockstep.

    windows 1..30   rollout 480 rows → critic trains, publishes nothing,
                    clears the rows itself and raises the gen gate;
                    the trainer never sees a row and the actor never steps
    window 31+      rollout → critic (score with the pre-update V, publish
                    `values`, then one value step) → trainer (GAE, then 5
                    optimiser steps) → HF weight export → SGLang reload → gate

Gate stream: `0` (trainer genesis, never offset) → `1..30` (critic cold start)
→ `31..` (trainer, offset by `critic_cold_start_windows`). The rollout runs
ungated (`pacing_window=None`) and is limited only by its in-flight cap of
`2 × BATCH_SIZE` samples, so it keeps submitting into a window's long tail
instead of idling on it; the gates still carry the weight version. In-flight
requests are aborted and resumed across each colocation hand-off — see
`_rollout_group`.

Update granularity is *derived*, not declared, so it is spelled out here:
`dp("batch") = dp_replicate × dp_shard = 2` (CP does not enter the batch mesh),
so `n_local = 480/2 = 240` and `mini_batch_size=48` gives
`n_mini = 5` — **5 optimiser steps per window, each over 48×2 = 96 samples**.
Changing `dp_shard_degree` silently changes the PPO update schedule. One
update per window would make clip-higher inert (with
`old_logprobs_source="rollout"` the policy has not moved inside the update, so
only the train/inference logprob gap — the thing TIS covers — reaches the
clip); five keeps §1's clip-higher load-bearing without shrinking an update to
a single prompt group.

The critic's own tempo: `max_tokens_per_micro` bounds the tokens resident on
the GPU per forward, *not* the update granularity — `engine/critic.py` feeds
the whole window to `train_value_accumulated`, so the critic takes
`value_epochs` steps per window. That is what makes §2's markers legible:
`warmup_steps=10` is 10 rollout steps, and "value_loss enters its normal range
after ~25 steps" is 25 windows.

The window is also split across the critic's DP mesh (`dp_shard_degree=2`), so
each rank fits 240 of the 480 rows and the two shards' gradients meet in FSDP's
reduce. Both halves of that — the DP split and the token-budget packing — come
from the same planner the actor uses (`backend/titan/plan.py`), and together
with dropping the critic's separate scoring forward (the value step's own
forward already runs on the pre-update weights, so it *is* the scoring pass)
they are why the critic is no longer the dominant term in a window's wall
clock.

Run length
----------

Nothing stops on `TrainerConfig.steps`: the trainer is TQ-driven and `steps`
only sets the LR-scheduler horizon, which `lr_decay_ratio=0.0` makes constant
anyway. The run ends when the rollout finishes `NUM_EPOCHS` over the dataset:
19,592 prompts ÷ 60 = 326 windows, minus 30 cold-start windows = **296 actor
steps**. §5 puts the useful window at ~170 steps with the peak at iter169, so
one epoch covers the peak with room to watch the decay past it. (The 32-prompt
tail of the epoch never fills a 480-row window and simply stays in the queue.)

Note that a second epoch would be a *verbatim* replay: `worker/rollout.py`
rebuilds the dataset per epoch from the same `seed`, so the shuffle is
identical.

Disk
----

Two independent things land on disk, on two budgets.

`CHECKPOINT` is the resumable one: every `interval` windows the actor writes a
DCP shard (model + optimizer + LR schedule + step), the critic writes its value
net + AdamW, and the rollout writes its data cursor, all under
`{runtime}/state/{service}/step-{N}/`. A run continues from the newest step all
three reached — see `meshy/service/checkpoint.py`. At fp32 that is ~27 GB per
replica, so ~54 GB per retained step and ~160 GB at `keep_latest_k=3`.

The other is the colocation weight-sync export — a full 2.25B HF checkpoint
written to `weights/actor_train-0/v{N}/` **every step**, because that is how
SGLang reloads the policy. It used to have no rotation anywhere in the tree;
over 296 steps that is terabytes, and §5 notes production jobs dying of
"临时存储超限". `TrainingServiceConfig.weights_keep_latest_k` now prunes it,
keeping the newest few plus any version a retained global checkpoint still
points at (a resumed inference server boots from exactly that path).

Not implemented
---------------

§1/§3 items this recipe cannot express, beyond the table above:

- **in-loop eval** is the one that matters most. §5's first boundary is that
  training reward keeps climbing past the peak and the ~3pp/100-step
  degradation is invisible to every training-side metric, so the peak ckpt can
  only be selected by in-loop eval. There is no eval role in `meshy/service/`,
  so this run produces no basis for choosing a checkpoint.
- **positive_lm_loss** and the three §2 gauges as dashboard metrics (they are
  loguru-only) are absent.
- **dynamic sampling** is implemented but off here; see
  `filter_zero_std_groups` in `_rollout_group` for why that is a choice rather
  than a gap.

What a resume does *not* recover: TransferQueue is RAM-only, so every row in
flight at the checkpoint is lost, and the rollout restarts from the prompt
cursor it had already handed out. Up to one pacing window of prompts is
therefore skipped rather than replayed — deliberate, so a resumed run never
trains twice on the same prompt.

See ``docs/critic_engine.md`` §4.

The backbone defaults to the Hub repo ``openbmb/MiniCPM5-2B-SFT``; set
``MINICPM5_LOCAL_PATH`` to a local checkout for ``HF_HUB_OFFLINE=1`` runs.

    S9_DATASET_PATH=/path/to/s9_no_advanced_math.jsonl \\
    HF_HOME=/path/to/hf-cache \\
    no_proxy="127.0.0.1,localhost,\\$no_proxy" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
        python scripts/launch.py --recipe recipe.justrl_ii_minicpm5_2b_128k
"""

from __future__ import annotations

import os
from pathlib import Path

from meshy.config import (
    GlobalCheckpointConfig,
    GAE_TRAINER_FIELDS_SHAPED,
    CriticServiceConfig,
    InferenceServiceConfig,
    RolloutServiceConfig,
    SamplingParams,
    TrainerConfig,
    TrainerParamsConfig,
    TrainingServiceConfig,
)
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor

#: Hub repo id by default; ``MINICPM5_LOCAL_PATH`` overrides it with a local
#: checkout (needed for ``HF_HUB_OFFLINE=1`` runs).
DEFAULT_MODEL = "openbmb/MiniCPM5-2B-SFT"
_LOCAL_MODEL = os.environ.get("MINICPM5_LOCAL_PATH")
MODEL_PATH = (
    str(Path(_LOCAL_MODEL).expanduser().resolve()) if _LOCAL_MODEL else DEFAULT_MODEL
)

try:
    DATASET_PATH = str(Path(os.environ["S9_DATASET_PATH"]).expanduser().resolve())
except KeyError as exc:
    raise RuntimeError("S9_DATASET_PATH must point to the S9 JSONL file") from exc

# ── §3: lengths and the actor/critic split ──────────────────────────────
SEQ_LEN = 131072
MAX_NEW_TOKENS = 126976
#: §1 soft-overlong penalty. The reward tapers linearly from 0 to
#: -OVERLONG_PENALTY_FACTOR over the last OVERLONG_BUFFER_LEN tokens of the
#: budget, so the model is pushed off the length cap before truncation
#: destroys the answer. Applied by the rollout as reward *shaping*, which is
#: why `external_advantage=True` no longer bypasses it (`meshy/reward.py`).
OVERLONG_BUFFER_LEN = 25395
OVERLONG_PENALTY_FACTOR = 1.0

NUM_CARDS = 8  # all 8 GPUs shared among inference, critic, and actor_train serially

# §3: 60 prompts × 8 samples = GBS 480.
ROLLOUT_BATCH = 60
GROUP_SIZE = 8
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # 480
NUM_EPOCHS = 1
#: LR-schedule horizon only -- nothing stops on it, and `lr_decay_ratio=0.0`
#: makes the schedule constant regardless. The real length is 296 actor steps
#: (one epoch, minus cold start); see "Run length" in the module docstring.
NUM_STEPS = 600

#: §2 cold start. 20, not the 30 this used to carry: 30 was budgeted for a
#: value head that started units away from its target and had to walk back
#: (run 20260918-022946 opened at value_loss=10.85 and burned ~16 windows just
#: getting under 1.0, then exited at 30/30 with var_reduction=-0.044 -- still
#: worse than no baseline). With the head zero-initialised the opening loss is
#: E[R^2] ~ 0.55 and the whole budget goes on learning, so the reference
#: implementation cut the same knob to 20 (miles phx_0711_ppo b83c97dc5,
#: measured out-of-warmup by step 17-18; `--critic-lr-warmup-iters 10` is
#: unchanged, and so is `warmup_steps` below). The env override exists so the
#: cold-start -> publish transition can be reached quickly in a smoke run.
COLD_START_WINDOWS = int(os.environ.get("XRL_COLD_START_WINDOWS", "20"))
CRITIC_SCORE_BATCH = int(os.environ.get("XRL_CRITIC_SCORE_BATCH", str(BATCH_SIZE)))

#: §3's "ckpt every 10 steps, rotated". The *same* instance goes to the
#: trainer, the critic and the rollout: the resume point is the newest window
#: all three committed a shard for, so a service on its own interval would
#: never intersect (``build_topology`` rejects that).
#:
#: Cost at ``dtype="float32"``: ~9 GB of parameters plus ~18 GB of AdamW
#: moments per replica, actor and critic both, so roughly 54 GB per retained
#: step -- ~160 GB resident at ``keep_latest_k=3``. That is on top of the
#: exported HF weights, which now rotate too (see "Disk" above).
#:
#: Resume with::
#:
#:     python scripts/launch.py --recipe recipe.justrl_ii_minicpm5_2b_128k \\
#:         --runtime-dir <the interrupted run's dir> --resume
CHECKPOINT = GlobalCheckpointConfig(
    enabled=os.environ.get("XRL_CHECKPOINT", "1") not in ("0", "false", ""),
    interval=int(os.environ.get("XRL_CHECKPOINT_INTERVAL", "10")),
    keep_latest_k=int(os.environ.get("XRL_CHECKPOINT_KEEP", "3")),
)


def _validate_dataset() -> None:
    if not Path(DATASET_PATH).is_file():
        raise FileNotFoundError(f"S9 dataset does not exist: {DATASET_PATH}")


def _sampling_params() -> SamplingParams:
    # §3: T=1.0, top_p=1.0.
    return SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens=MAX_NEW_TOKENS
    )


def _actor_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2B",
        seq_len=SEQ_LEN,
        steps=NUM_STEPS,
        lr=1e-6,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.98,
        # §3: adam eps 1e-8. Already the dataclass default; stated so the
        # recipe item is checkable against this file.
        eps=1e-8,
        warmup_steps=0,
        # §3: constant 1e-6.
        lr_decay_ratio=0.0,
        # fp32 master weights; bf16 matmuls (mixed_precision_param default).
        dtype="float32",
        compile_model=True,
        # 32k tokens/rank x 42 layers does not fit next to the 130k-vocab LM
        # head under per-op SAC.
        activation_checkpoint_mode="full",
        # 8 GPUs: CP4 × FSDP2.
        dp_shard_degree=2,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=4,
        enable_checkpoint=True,
        checkpoint_folder="checkpoint",
        dump_folder="./outputs/justrl_ii_minicpm5_2b_128k",
    )


def _critic_trainer_config() -> TrainerConfig:
    """Same architecture and parallel layout as the actor, on the shared cards.

    The critic is a separate replica: own FSDP mesh, own optimiser, no shared
    parameter. CP4 for the same reason the actor uses it -- 128k does not fit
    on one rank -- and because GAE walks adjacent timesteps, `predict_vapo_gae`
    gathers across the CP group (`cp.py::gather_seq`), which is exactly the
    path `scripts/smoke/critic_cp_gae.py` verifies.
    8 GPUs: CP4 × FSDP2.
    """
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2B",
        seq_len=SEQ_LEN,
        max_norm=1.0,
        steps=NUM_STEPS,
        # fp32 master weights; bf16 matmuls (see ``critic_coldstart_32k.py``).
        dtype="float32",
        compile_model=False,
        dp_shard_degree=2,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=4,
        # train_backbone=True keeps every layer's activations alive for the
        # backward, so the critic needs full checkpointing regardless.
        activation_checkpoint_mode="full",
        enable_checkpoint=False,
        dump_folder="./outputs/justrl_ii_minicpm5_2b_128k/critic",
    )


def _trainer_params() -> TrainerParamsConfig:
    return TrainerParamsConfig(
        # 480 rows / dp(2) = 240 local rows, so this yields 5 optimiser steps
        # per window over 96 samples each. See "Run tempo" in the module
        # docstring -- this number, not `batch_size`, sets the PPO update
        # schedule, and it moves when `dp_shard_degree` does.
        mini_batch_size=48,
        micro_batch_size=1,
        ppo_clip_eps_low=0.2,
        ppo_clip_eps_high=0.28,
        old_logprobs_source="rollout",
        calculate_per_token_loss=True,
        # §1 TIS: the train/inference logprob mismatch fuse. Near-zero load in
        # practice (tis ~ 1.000).
        use_tis=True,
        tis_ratio_min=0.5,
        tis_ratio_max=5.0,
        logprob_chunk_size=2048,
        # §1 VAPO-GAE, run here rather than in the critic. The critic publishes
        # V per token; the trainer walks
        #   delta[t] = R[t] + gamma*V[t+1] - V[t]
        #   A[t]     = delta[t] + gamma*lambda*A[t+1],  lambda = 1 - 1/(alpha*L)
        # and feeds A straight into the PPO ratio *per token*. Reducing it to
        # one scalar per sequence (the critic's `publish_mode="advantage"`)
        # would throw away exactly the within-sequence credit assignment the
        # value baseline exists to provide.
        enable_gae=True,
        gae_gamma=1.0,
        # §1: alpha must match the response length; the paper's 0.05 collapses
        # the credit half-life to ~500 tokens at 15k. Same value the critic's
        # own GAE used, and it has to stay in sync with `CriticServiceConfig`
        # only in spirit -- the trainer is the one that runs the recursion now.
        gae_alpha=1.5,
    )


def _inference_group() -> ServiceGroup:
    # 8 SGLang instances at TP1, one per GPU.
    #
    # Concurrency: the rollout caps itself at `2 * BATCH_SIZE` = 960 samples in
    # flight, i.e. 120 per instance against `max_running_requests=64`. The
    # surplus queues inside SGLang, which is the intended behaviour -- a queued
    # request holds no KV, and the 64 that do run get 1_200_000 / 64 = 18.75k
    # tokens of budget each. Raise `max_running_requests` only if the queue is
    # demonstrably starving the GPU; lower it if the logs show frequent
    # retract/preempt. Do not raise `mem_fraction_static`, which the colocation
    # hand-off has to release and restore every window.
    #
    # Why 0.80 and not the 0.88 this used to carry: `offload_to_cpu()` +
    # `empty_cache()` on the trainer and the critic do *not* return the whole
    # card. The CUDA context, the FSDP2/CP4 NCCL buffers and the cuBLAS
    # workspaces stay resident for the life of the process, and on
    # 20260918-022946 that was **4.33 GiB (actor_train rank 0) + 7.33 GiB
    # (critic rank 0) = 11.7 GiB** still pinned on GPU 0 while SGLang thought
    # it owned the card. 0.88 x 79.18 = 69.7 GiB budget against 79.18 - 11.7 =
    # 67.5 GiB actually available, so SGLang ran with no headroom at all and
    # died at 20:02:45 on a 1018 MiB prefill input-logprob chunk (2048 tok x
    # 130560 vocab x fp32) with 927 MiB free -- taking the colocation manager
    # and the whole 8-card job with it.
    #
    #   0.80 x 79.18 = 63.3 GiB budget
    #   + 11.7 GiB colocation residency = 75.0 GiB of 79.18  ->  ~4 GiB spare
    #
    # `max_total_tokens` comes down with it, because the two have to agree:
    # KV is 2 kv_heads x 128 head_dim x 42 layers x 2 (K+V) x 2 B = 43008
    # B/token, so 1_440_000 tokens is 57.7 GiB of KV and, with ~8.9 GiB of
    # non-KV residency measured on that run, 66.6 GiB total -- i.e. the *token
    # cap*, not the fraction, was what actually sized the pool. SGLang clamps
    # `min(profiled, max_total_tokens)` (`kv_cache_configurator.py::
    # _apply_token_constraints`) and only warns, so leaving 1_440_000 here
    # would silently hand the pool back to the profiler and make this comment
    # a lie. 1_200_000 x 43008 B = 51.6 GiB of KV, 60.5 GiB total.
    #
    # The cost is per-request budget: 22.5k -> 18.75k tokens against a p50
    # response of ~10k and a mean of ~18k. That run was already retracting at
    # token usage 1.00, so expect retract/re-prefill to get *more* frequent,
    # not less. Retraction is graceful and OOM is not, so this is the right
    # side to err on -- but if the churn dominates the window, drop
    # `max_running_requests` to 48 (25k each) before touching either number
    # here.
    return ServiceGroup(
        id="actor_infer",
        n_replicas=NUM_CARDS,
        n_gpus_per_replica=1,
        config=InferenceServiceConfig(
            model_path=MODEL_PATH,
            server_args={
                "model_path": MODEL_PATH,
                "tp_size": 1,
                "attention_backend": "fa3",
                "mem_fraction_static": 0.80,
                "max_running_requests": 64,
                "max_total_tokens": 1200000,
                "schedule_conservativeness": 1.2,
                "enable_memory_saver": True,
            },
        ),
    )


def _training_group() -> ServiceGroup:
    return ServiceGroup(
        id="actor_train",
        n_replicas=1,
        n_gpus_per_replica=NUM_CARDS,
        colocate_with="actor_infer",
        wait_until=["actor_infer"],
        config=TrainingServiceConfig(
            model_path=MODEL_PATH,
            trainer_config=_actor_trainer_config(),
            trainer_params=_trainer_params(),
            batch_size=BATCH_SIZE,
            # `advantage` is gone: the critic publishes `values` and this
            # trainer builds the advantage itself. `values` is now the column
            # whose absence holds a row back until the critic has scored it,
            # and `raw_reward` carries the pre-shaping reward for the metrics.
            tq_fields=list(GAE_TRAINER_FIELDS_SHAPED),
            timer_enabled=True,
            # Keeps the rollout's gate stream monotonic across the hand-off
            # from the critic (which paces during cold start) to the trainer.
            critic_cold_start_windows=COLD_START_WINDOWS,
            checkpoint=CHECKPOINT,
        ),
    )


def _critic_group() -> ServiceGroup:
    return ServiceGroup(
        id="critic",
        n_replicas=1,
        n_gpus_per_replica=NUM_CARDS,
        # Colocated with actor_infer on the same 8 cards. The pipeline is
        # serial (rollout → critic → train), so there is no concurrent GPU
        # pressure: enable_memory_saver on SGLang releases the KV-cache
        # budget before the critic or trainer takes the node.
        colocate_with="actor_infer",
        wait_until=["actor_infer"],
        config=CriticServiceConfig(
            # Backbone from the *base* model, never the actor's live weights.
            model_path=MODEL_PATH,
            trainer_config=_critic_trainer_config(),
            score_batch_size=CRITIC_SCORE_BATCH,
            # Publish V(s_t) per token and let the trainer run GAE. The critic
            # still owns the value net and the scoring order (score with the
            # pre-update V, publish, then take the value step); it just no
            # longer collapses the result to one number per sequence.
            publish_mode="values",
            # Tokens per critic forward (rows x padded row length), not rows.
            # The window's rows span two orders of magnitude -- p50 ~10k, p90
            # ~40k, p99 ~95k on this dataset -- so a fixed row count is sized
            # for the longest row and leaves the batch dimension nearly empty
            # for the other 90%. With a budget the planner sorts by length and
            # packs to it: short rows share a forward, a 95k row still gets one
            # to itself. On this run's length distribution that is 480 forwards
            # per window -> 73 per rank.
            #
            # The budget cannot raise peak activation memory above what the
            # run already survived: a row longer than the budget is placed
            # alone regardless, so the peak is set by the longest *row*
            # (126976, i.e. 32k per rank under CP4), not by the budget. 65536
            # is 16k per rank -- half that.
            #
            # This bounds memory only -- the value step still accumulates over
            # the whole window and takes a single optimiser step.
            max_tokens_per_micro=65536,
            # §2: a value-head-only critic is a linear probe on frozen
            # features and plateaus before it beats the whitening baseline.
            train_backbone=True,
            lr=5e-6,
            # §2: 10-iter lr warmup absorbs the cold-start value_loss spike
            # (peak ~32) without moving the head a long way on it. The critic
            # takes one step per window, so this is 10 rollout steps -- the
            # unit §2 counts in.
            warmup_steps=10,
            cold_start_windows=COLD_START_WINDOWS,
            # Optimiser steps per window. 1 keeps the critic's clock equal to
            # the rollout's, which is what makes §2's "~25 steps to normal
            # range" and "end near 0.4" readable off the window log.
            value_epochs=1,
            # Return discount for the value *target*: with gamma=1 and a single
            # outcome reward, the target is R broadcast across the response,
            # i.e. §1's [R, R, ..., R]. Keep it equal to the trainer's
            # `gae_gamma` -- a value net fitted to one discount and consumed
            # under another is biased everywhere but the terminal token.
            gamma=1.0,
            # Only used when `publish_mode="advantage"`; the trainer's
            # `gae_alpha` is what governs the live advantage.
            alpha=1.5,
            max_norm=1.0,
            checkpoint=CHECKPOINT,
        ),
    )


def _rollout_group() -> ServiceGroup:
    return ServiceGroup(
        id="rollout",
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_train", "actor_infer", "critic"],
        config=RolloutServiceConfig(
            model_path=MODEL_PATH,
            dataset="meshy.dataset.s9_math:S9Math",
            dataset_kwargs={"path": DATASET_PATH, "batch_size": ROLLOUT_BATCH, "seed": 42},
            reward="meshy.dataset.s9_math:S9Math.reward",
            # §1 overlong soft punish. Shaping is applied to the reward here,
            # so `reward` on the wire is R (what the critic regresses onto and
            # what GAE consumes) and `raw_reward` keeps the unshaped 0/1 the
            # solve-rate metrics need. This is the piece `external_advantage`
            # used to skip, because the penalty lived in the advantage
            # pipeline.
            reward_shaping="meshy.reward:soft_overlong_penalty",
            reward_shaping_kwargs={
                "max_response_len": MAX_NEW_TOKENS,
                "buffer_len": OVERLONG_BUFFER_LEN,
                "penalty_factor": OVERLONG_PENALTY_FACTOR,
            },
            sampling_params=_sampling_params().as_dict(),
            group_size=GROUP_SIZE,
            num_epochs=NUM_EPOCHS,
            poll_interval=2.0,
            # Keep the accelerators fed instead of draining one window to zero
            # before starting the next. With `pacing_window=1` the rollout
            # dispatches exactly `BATCH_SIZE` samples and then goes idle until
            # the gate, so the whole window waits on its slowest sequence --
            # at `max_new_tokens=126976` that tail is most of the wall clock,
            # and it is spent with SGLang nearly empty.
            #
            # `None` drops the gate-derived budget entirely (the genesis gate is
            # still awaited, so generation never precedes the first weight
            # version); the in-flight cap below is then the only limiter, and
            # the rollout keeps submitting while a previous window drains.
            pacing_window=None,
            # Max samples in flight, in *samples* -- `_run_rollouts` divides by
            # `group_size` for its group semaphore. 2 windows keeps one window
            # generating while the other is being scored/trained.
            #
            # Backpressure does not disappear with `pacing_window=None`: the
            # serial ring releases SGLang whenever the critic or trainer takes
            # the node, so generation is suspended rather than racing ahead.
            # In-flight requests survive that -- `release_for_colocate` aborts
            # them and the client resumes each one from `prompt + partial`
            # (§3 "partial rollout"). The cost is a re-prefill of the partial
            # output per preemption, and rows up to ~2 versions stale by the
            # time the trainer consumes them, which is what TIS covers. Set
            # `pacing_window=2` to put the gate-anchored bound back.
            async_max_running_request=2 * BATCH_SIZE,
            # The critic owns the advantage: the rollout neither computes nor
            # writes the column, which is what makes the trainer wait for it.
            # This is also what rules the overlong penalty out -- it forces
            # `advantage_fn = None` and skips `meshy/advantage.py` entirely.
            external_advantage=True,
            # §1 dynamic sampling. Off by default *here* and not an oversight:
            # with a value baseline a zero-variance group still carries a
            # gradient (recipe §1 is why PPO replaced GRPO), so dropping one
            # trades data for compute rather than rescuing a dead batch.
            #
            # Turning it on gives §1's arrangement: the filter drops
            # all-pass/all-fail groups (judged on `raw_reward`, so the length
            # penalty cannot fake variance) and `oversample_factor` bounds how
            # far the epoch may over-run replacing them. Replacement is
            # implicit -- `pacing_window=None` keeps prompts in flight behind
            # the dropped one and every consumer takes a fixed row count off
            # TQ, so a window still fills; the factor only stops a stretch of
            # uniformly-solved prompts from burning the dataset.
            filter_zero_std_groups=False,
            oversample_factor=2.0,
            checkpoint=CHECKPOINT,
        ),
    )


def build_service_groups() -> list[ServiceGroup]:
    return [_inference_group(), _training_group(), _critic_group(), _rollout_group()]


SERVICE_GROUPS = build_service_groups()
COLOCATIONS = [
    ColocationRing(
        group_id="actor_card",
        ring=(
            ("actor_infer", SchedulingMode.FALLBACK),
            ("critic", SchedulingMode.ON_DEMAND),
            ("actor_train", SchedulingMode.ON_DEMAND),
        ),
    )
]


def main() -> None:
    _validate_dataset()
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
