# Meshy Architecture: Service, Engine, Worker, and TransferQueue

> This document describes the current Meshy architecture. Every role is a
> Service, data always flows through TransferQueue (TQ), and data availability
> provides the control signal. Colocated GPU ownership is passed through TQ as
> a token ring; see [`colocation.md`](colocation.md). HTTP is reserved for
> SGLang generation and management endpoints.

## 1. Overview

Meshy runs one card-level SPMD `torchrun`: each GPU has an ignitor process that
starts all Services assigned to that card.

| Service | Role | Implementation | GPU | Responsibility |
|---|---|---|---|---|
| Inference | `inference` | `meshy/service/inference.py:SGLangService` | yes | SGLang generation and hot weight updates |
| Training | `training` | `meshy/service/training.py:TitanTrainingService` | yes | Consume training batches, run steps, save HF checkpoints, publish weights and gates |
| Rollout | `rollout` | `meshy/service/rollout.py:RolloutService` | no | Hold the dataset, generate responses, compute rewards/advantages, and write samples to TQ |

```text
Service   (meshy/service)  lifecycle, placement, readiness, and colocation wiring
  └── Engine  (meshy/engine)   model, process group, GPU residency, checkpoints; knows nothing about TQ
        └── Worker (meshy/worker)   TQ contract and role loop; runs on the replica master
```

The rollout worker calls SGLang `/generate` and puts the GRPO columns plus
`weight_version` into `data.train`. Titan consumes rows with an AND filter,
trains, and publishes one `gen_gate` row per synchronized version. In a
colocated layout, a request ledger at
`control.colocate.<group>.request` transfers the GPU token and carries the
checkpoint path used to reload inference weights.

**Boundaries:** sample data, generation gates, and resource arbitration use
TQ. SGLang's native HTTP API, wrapped by `SGLangEngine`, handles generation and
inference memory/weight management.

## 2. Service Layer (`meshy/service/`)

| Concept | File | Responsibility |
|---|---|---|
| `GPU` | `base.py` | Host and global/node/local rank identifiers |
| `ServiceGroup` | `base.py` | Declarative DAG node with typed config, replica/card shape, `colocate_with`, and `wait_until` |
| `Service` | `base.py` | Per-card runtime: `from_info`, `ignite`, readiness, and colocation manager construction |
| `ServiceInfo` / `Topology` | `topology.py` | Deterministically derives endpoints, ports, card ownership, and ring membership |
| `build_topology` | `topology.py` | Allocates card blocks in declaration order, reuses blocks for colocation, derives ports, and validates collisions |
| `resolve_service` | `registry.py` | Resolves config instances through their `service_cls` class attribute; string roles remain compatible |
| `Ignitor` | `ignite.py` | Discovers GPUs, gates startup on dependencies, ignites Services, issues genesis, starts CPU Services, and joins children |
| `SpmdService` | `spmd.py` | Shared GPU Service template; colocated groups must belong to a ring |
| `SGLangService` | `inference.py` | Starts `sglang.launch_server` children |
| `TitanTrainingService` | `training.py` | Wires `TitanEngine` and `TitanWorker` |
| `RolloutService` | `rollout.py` | CPU-only process running `RolloutWorker.run_async()` |
| `ColocationManager` / `NoopColocationManager` | `colocation.py` | Token-ring state machine or no-op outside a ring |
| `RuntimeDir` | `runtime.py` | Weights, logs, TensorBoard, trajectory artifacts, and readiness markers |
| Bootstrap store | `bootstrap.py` | Run-level `TCPStore`, hosted by rank 0 unless `XRL_BOOTSTRAP_ADDR` is provided |

### 2.1 Typed configuration (`meshy/config.py`)

Each role is a `ServiceConfig` subclass. Its class attributes are the plugin
contract:

```python
@dataclass
class ServiceConfig:
    role: ClassVar[str]
    service_cls: ClassVar[str]           # "package.module:ClassName"
    uses_gpu: ClassVar[bool] = True
    endpoint_port_base: ClassVar[int | None]
    dist_port_base: ClassVar[int | None]
```

`InferenceServiceConfig` uses port bases 30000/40000 and has `model_path` and
`server_args`. `TrainingServiceConfig` uses 31000/41000 and contains Titan
configs, `batch_size`, `weight_sync_mode`, `stream_minibatch`, `tq_fields`,
`partition_id`, and polling settings. `RolloutServiceConfig` is CPU-only and
contains dataset, reward, advantage, sampling, grouping, concurrency, pacing,
filtering, and trajectory settings.

`ServiceGroup.role` is derived from the config class. `from_info` receives the
complete config, so recipe fields are not dropped. A new role needs one config
subclass and one Service subclass; no registry edit is necessary.

### 2.2 `SpmdService` child wiring

```text
_run_runtime()  (one spawned child per card)
  redirect_output(<log_dir>/<name>-<rank>.log); die_with_parent()
  engine = build_engine(); engine.init()            # independent process group
  colocation = build_colocation_manager(...)       # Noop outside a ring
  if engine.is_master: colocation.start()
  worker = build_worker(engine, colocation)
  if not master: engine.run_command_loop(); return
  worker.tq_thread(); readiness_event.set()
  engine.run_command_loop()
```

Non-master readiness returns immediately. The master waits for the child event,
checks liveness, and publishes `runtime.mark_ready(name)`.

### 2.3 `SGLangService`

Only a replica's node-local leader calls `Popen` for
`sglang.launch_server`. Multi-node launches add `--nnodes`, `--node-rank`, and
`--dist-init-addr`; torchrun variables are removed and CUDA visibility is
remapped. The master polls `/health_generate`. In a colocated layout it pauses
and drains requests, releases KV cache and weights, starts the ring manager,
and publishes readiness only after the card is free.

## 3. Engine Layer (`meshy/engine/`)

`SpmdEngine` creates an independent process group, uses a separate gloo group
for command broadcasts/barriers, and runs `execute(payload, samples)` on all
ranks. Rank 0 submits commands; non-masters stay in the command loop.

`TitanEngine` builds the Titan trainer, supports `step`, acquire, and release
commands, offloads colocated models to CPU, saves HF checkpoints, and updates
disaggregated SGLang through `/update_weights_from_disk`. `weight_sync_mode`
currently resolves to disk.

`SGLangEngine` is an HTTP client. It rotates inference endpoints, requests log
probabilities, resumes aborted generations with their prefix, and exposes
readiness, idle, pause/continue, memory, and weight-loading operations. A
non-genesis colocation grant must include `payload_ref`.

## 4. Worker Layer (`meshy/worker/`)

`TQWorker` declares `TQInput` and `TQOutput`, fetches rows with an AND filter,
converts them with `adapter.td_to_samples`, calls `process_tq_batch`, writes
outputs, optionally clears consumed rows, and retries each phase independently.
Producer clients use a single-thread executor because TQ clients are not
thread-safe.

`TitanWorker` consumes `tq_fields` as consumer `titan`, clears successful rows,
and emits `gen_gate` at startup and after each synchronized step. `RolloutWorker`
rebuilds the dataset each epoch, generates grouped responses, computes reward
and advantage, and writes `GRPO_FIELDS` to `data.train`. Its
`pacing_window` consumes gates; `async_max_running_request` limits concurrent
groups.

## 5. Data Plane and Column Contract

The default `data.train` partition is reused for the whole run. The trainer
calls `clear_samples(meta)` after consumption. A row is eligible only when all
requested columns exist, and `task_name` isolates consumers.

| Column | Storage | Producer | Meaning |
|---|---|---|---|
| `tokens`, `logprobs`, `mask_assistant` | jagged nested | rollout | Full sequence, rollout log probabilities, and loss mask |
| `advantage`, `reward` | scalar f32 | rollout | Group advantage and raw reward |
| `weight_version` | scalar i64 | rollout | Version used for generation; a lower bound when `W > 1` |
| `truncated`, `repetition`, `mixed_version` | scalar i64 | rollout | Generation quality flags |

Unknown columns use `NonTensorStack`. `td_to_samples` clones values so samples
do not alias the packed batch buffer. Detailed per-generation fields remain in
`trajectories.jsonl`; trainer metrics are computed on rank 0 from the global
batch.

## 6. Generation Gates and Pacing

Training emits `gate_0(v0)` at startup and one `gate_N(vN)` after every
synchronized step. Gates are rows in `gen_gate` carrying `gate_step` and
`weight_version`; consumers read and clear them.

| `pacing_window` | Meaning |
|---|---|
| `1` or `"auto"` | Lock-step generation |
| `N >= 2` | Bounded off-policy overlap |
| `None` | No budget after the initial gate; gates still refresh version state |

After `g` gates, rollout may start `(g - 1 + W) × batch_size` samples. No mode
generates before the first gate establishes the inference version.

## 7. Colocation Token Ring

Declare `ColocationRing` with ring members such as inference `FALLBACK` and
training `ON_DEMAND`. `build_topology` verifies that members share the same
cards. The releasing service frees memory before publishing a grant; the
acquiring service restores memory before waking its Worker. Training grants
carry the checkpoint path as `payload_ref`. Disaggregated services use
`NoopColocationManager` and the same Worker code.

## 8. One Lock-Step Step (W=1)

```text
Ignitor: start SGLang → ready → release memory → marker
Ignitor: start Titan after the marker → setup and offload → marker
Ignitor: barrier → issue_genesis → SGLang restores initial weights
Rollout: consume gate_0 → generate at v0 → put(data.train)
Titan: request GPU → train → save v1 → release(payload_ref=v1)
SGLang: load v1 and continue; Titan clears rows and emits gate_1(v1)
```

In a disaggregated layout, GPU hand-off is absent and rank 0 posts the new
checkpoint to inference. With `stream_minibatch`, synchronization waits until
`batch_size` samples have accumulated.

## 9. Startup and TQ Deployment

`scripts/launch.py` resolves the runtime directory, imports the recipe to count
cards, hosts bootstrap, starts TQ, and launches one torchrun. Unless
`TRANSFER_QUEUE` is exported, capacity is derived as
`max(4 × batch, 2 × max_running + batch, 1024)`. `XRL_TQ_ENDPOINTS` may point
to an externally managed endpoint file. Direct `torchrun -m recipe.x` can host
bootstrap on rank 0, but TQ must be started separately.

## 10. Test Map

| Test | Coverage |
|---|---|
| `tests/test_service_plugin.py` | Config plugin and service resolution |
| `tests/test_meshy_colocation.py` | Token ledger and ring priority |
| `tests/test_bootstrap.py` | Bootstrap hosting and client resolution |
| `tests/test_runtime_paths.py` | Runtime directory overrides |
| `tests/test_service_logging.py` | Child log redirection |
| `tests/test_tq_spec.py` | Capacity derivation and endpoint discovery |
| `tests/test_tq_gen_gate.py` | Gate ordering and bounded state |
| `tests/test_tq_grpo_pipeline.py` | GRPO columns, staleness, batching, and index recycling |
| `tests/test_agentloop_driver.py` | End-to-end rollout driver |
| `tests/test_rollout_advantage.py`, `test_math_reward.py` | Advantages and math rewards |
| `tests/test_weight_sync.py` | Checkpoint grants and weight synchronization |
| `tests/test_dynamic_batching*.py` | Dynamic batching layouts |
| `tests/test_minicpm5_*.py` | MiniCPM5 registration and parity |
| `scripts/smoke/run.sh` | GPU smoke tests |

## 11. Future Directions

- Per-rank direct TQ consumption instead of rank-0 broadcast.
- Multiple rollout replicas with sharded generation.
- Exact per-sample staleness labels when `W > 1`.
- Zero-copy weight transfer.
- Reclaim closed colocation ledger rows.
- Activate `hf_save_interval`.
