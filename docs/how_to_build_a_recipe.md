# How to Build a Recipe

A Meshy recipe is a Python module that declares the services in an RL run,
their configuration, and their placement. Building one involves three steps:
load a dataset, choose or implement services, and assemble them into
`SERVICE_GROUPS` plus an optional `COLOCATIONS` ring.

This guide uses GRPO as the running example. It assumes the environment in
the [Quick Start](../README.md#quick-start) is installed and commands run from
the repository root. See [grpo_gsm8k.py](../recipe/grpo_gsm8k.py) for an existing
recipe with configurable GPU placement.

## 1. Import a Dataset

### Use an existing adapter

The rollout service loads `dataset` from a `"module:Class"` string and passes
`dataset_kwargs` to its constructor. These adapters are available:

| Adapter | Data source | Reward function |
|---|---|---|
| `meshy.dataset.gsm8k:GSM8K` | Hugging Face `openai/gsm8k`, `main` configuration | `meshy.dataset.gsm8k:GSM8K.reward` |
| `meshy.dataset.math:MATH` | Hugging Face `BytedTsinghua-SIA/DAPO-Math-17k` | `meshy.dataset.math:MATH.reward` |
| `meshy.dataset.s9_math:S9Math` | Local JSONL with `prompt` message lists and a scalar `label` | `meshy.dataset.s9_math:S9Math.reward` |

For example, configure GSM8K as follows:

```python
from meshy.config import RolloutServiceConfig

rollout_config = RolloutServiceConfig(
    model_path="Qwen/Qwen3-1.7B",
    dataset="meshy.dataset.gsm8k:GSM8K",
    dataset_kwargs={"batch_size": 4, "split": "train", "seed": 42},
    reward="meshy.dataset.gsm8k:GSM8K.reward",
    group_size=4,
    sampling_params={"temperature": 1.0, "max_new_tokens": 1024},
    pacing_window=1,
)
```

Here, a dataset batch contains four **prompts**. With `group_size=4`, it
produces 16 **training samples** before any filtering or generation failures.
`TrainingServiceConfig.batch_size` counts these generated samples.

GSM8K and MATH accept `hf_kwargs`, which override arguments to
`datasets.load_dataset`, including `path`, `split`, and `data_files`. The
replacement data must still match the adapter's expected columns. For S9Math,
pass `path` in `dataset_kwargs` or set `S9_DATASET_PATH`; its
`prompt_max_tokens` option checks tokenized prompt length.

### Write an adapter for your own data

The dataset contract is small:

1. The constructor accepts the configured `dataset_kwargs`.
2. `next_batch(builder)` returns a list of prompt `Sample` objects.
3. An empty list signals the end of the epoch.

`RolloutWorker` constructs a new dataset instance at the start of every epoch.
Reset iteration state in the constructor. A fixed shuffle seed therefore
produces the same ordering each epoch.

Use [SampleBuilder](../meshy/utils/sample.py) to apply the model's chat template
and initialize aligned tokens, log probabilities, and loss masks. Put the
reference answer in `sample.ground_truth`; the rollout worker appends the
generated assistant response before calling your reward function.

For a JSONL file with records such as
`{"question": "What is 2 + 2?", "answer": "4"}`, create an importable module
such as `recipe/my_dataset.py`:

```python
from datasets import load_dataset

from meshy.utils.sample import Sample, SampleBuilder


class MyDataset:
    def __init__(self, path: str, batch_size: int, seed: int = 42):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.data = load_dataset(
            "json", data_files=path, split="train"
        ).shuffle(seed=seed)
        self.batch_size = batch_size
        self.index = 0

    def next_batch(self, builder: SampleBuilder) -> list[Sample]:
        if self.index >= len(self.data):
            return []
        end = min(self.index + self.batch_size, len(self.data))
        samples = []
        for index in range(self.index, end):
            row = self.data[index]
            sample = builder.build_sample([
                {
                    "role": "user",
                    "content": row["question"] + "\nReturn only the final answer.",
                }
            ])
            sample.ground_truth = str(row["answer"]).strip()
            samples.append(sample)
        self.index = end
        return samples

    @staticmethod
    def reward(sample: Sample) -> float:
        response = sample.messages[-1]["content"].strip()
        return float(response == sample.ground_truth)
```

Connect it with `dataset="recipe.my_dataset:MyDataset"`,
`dataset_kwargs={"path": "/data/train.jsonl", "batch_size": 4, "seed": 42}`,
and `reward="recipe.my_dataset:MyDataset.reward"`. This reward uses exact
text matching; replace it with the scoring rule appropriate for your task.
Always configure `reward` explicitly.

Subclassing [Dataset](../meshy/dataset/base.py) is also supported: its constructor
loads Hugging Face data, and subclasses implement `apply_chat_template`.
Its current `next_batch` stops when `index + batch_size >= len(dataset)`, so
it skips both a partial tail and a final batch that exactly reaches the end.
The standalone adapter above shows explicit tail handling. A partial prompt
batch can still leave fewer samples than the trainer's fetch threshold.

### Choose the advantage function

By default, the worker normalizes rewards within each prompt's response group
using `meshy.worker.rollout:grpo_advantage`. A group of one uses the reward
directly as its advantage.

Set `RolloutServiceConfig.advantage` to a callable or a `"module:function"`
string to override this behavior. The function receives the completed group
and `advantage_kwargs`. It may either set each `sample.advantage` in place and
return `None`, or return one numeric advantage per sample. Reward functions
are synchronous callables taking one completed sample and returning a scalar.

For a more specialized implementation, see the length and overlong reward
shaping in [meshy/advantage.py](../meshy/advantage.py) and its configuration in
[math_grpo_minicpm5_2b.py](../recipe/math_grpo_minicpm5_2b.py).

## 2. Write a Service

### Understand the boundaries

A service owns process startup, placement, readiness, and the wiring between
an engine and a worker. The engine owns computation and model state. The
worker owns the role's business loop and TransferQueue (TQ) input/output
contracts.

```text
Service
  +-- Engine: model operations, process groups, GPU residency, checkpoints
  +-- Worker: queue reads/writes, batching, pacing, role-specific decisions
        +-- calls the Engine
```

Samples, generation gates, and GPU ownership requests travel through TQ.
Generation and inference management use SGLang's native HTTP endpoints.
Engines do not own TQ clients.

### Reuse the built-in services

Most new datasets and reward functions only require new configuration:

| Config in `meshy.config` | Service | Worker and engine | Main settings |
|---|---|---|---|
| `InferenceServiceConfig` | `meshy.service.inference:SGLangService` | Launches SGLang server processes; uses `SGLangEngine` for management, with no TQ worker | `model_path`, `server_args` |
| `TrainingServiceConfig` | `meshy.service.training:TitanTrainingService` | `TitanWorker` + `TitanEngine` | `trainer_config`, `trainer_params`, `batch_size`, `stream_minibatch`, `partition_id`, `tq_fields` |
| `RolloutServiceConfig` | `meshy.service.rollout:RolloutService` | `RolloutWorker` + `SGLangEngine` HTTP client | Dataset, reward, advantage, sampling, concurrency, pacing |

The rollout service requires no GPU of its own. It discovers inference
endpoints from the topology and reads the first training service's batch size
for pacing. Start with one trainer replica and one rollout replica; simply
adding replicas does not provide dataset sharding or independent gate streams.

Available worker building blocks:

| Worker | Use |
|---|---|
| [Worker](../meshy/worker/base.py) | Base thread lifecycle: implement `run()`, use `start()`, `stop()`, and `join()` |
| [TQWorker](../meshy/worker/tq.py) | Queue consumer or producer with column contracts, client lifecycle, retries, and control polling |
| [RolloutWorker](../meshy/worker/rollout.py) | Dataset iteration, grouped generation, reward, advantage, trajectory logging, and writes to the training partition |
| [TitanWorker](../meshy/worker/titan.py) | Fetch training batches, acquire GPUs when colocated, call the training engine, and publish generation gates |

Available engine building blocks:

| Engine | Use |
|---|---|
| [SGLangEngine](../meshy/engine/sglang.py) | Async `generate(input_ids, sampling_params=...)`, endpoint rotation, continuation after interrupted generation, and weight/memory management |
| [TitanEngine](../meshy/engine/titan.py) | Distributed policy training through `step(samples, sync=...)`, checkpoint export, weight synchronization, and CPU/GPU offloading |
| [SpmdEngine](../meshy/engine/spmd.py) | Base for a new distributed engine; implement `setup()` and `execute(payload, samples)` and dispatch work with `submit_command()` |

`SGLangEngine` is a client, so constructing one does not start an inference
server. `SGLangService` owns that process. For distributed training,
`SpmdService` starts an engine on every card in the replica, runs the worker
loop only on the replica master, and runs the engine command loop on all ranks.

### Define a custom worker's queue contract

Subclass `TQWorker`, call `configure_tq(...)` in its constructor, and implement
`process_tq_batch(samples)`. Declare:

- `TQInput(partition, fields, batch_size, consumer, clear_after_success)` for
  the primary input. A row is eligible only when **all** requested columns are
  present. `consumer` identifies the logical consuming task.
- Named `TQOutput(fields, new_rows, partition)` entries for outputs.
  `new_rows=True` requires a destination partition; `new_rows=False` writes
  columns back to the fetched rows and must omit the partition.
- Optional named control inputs through `controls`, and initial outputs
  through `startup_tq_outputs()`.

The processing method receives per-sample `TensorDict` objects, rather than
the dataset's `Sample` objects, and returns a mapping from output names to
batched `TensorDict` values. Use
[adapter.samples_to_td](../meshy/transferqueue/adapter.py) to pack outputs.
Set `clear_after_success=True` only at the final consumer of rows: clearing
an intermediate stage's rows removes data needed downstream. Producer-only
workers can use `open_tq()`, `write_tq_output()`, and `read_tq_control()` in an
async loop; see `RolloutWorker.run_async()` for client setup and cleanup.

The built-in GRPO producer and trainer agree on `GRPO_TRAINER_FIELDS`:

```text
tokens, logprobs, mask_assistant, advantage, weight_version,
reward, truncated, repetition, mixed_version
```

`tokens`, `logprobs`, and `mask_assistant` describe the aligned full sequence;
the remaining columns are per-sample scalars. Both services default to the
`data.train` partition. Changing `tq_fields` requires a producer for every
requested field, or the trainer will keep waiting. A separate scoring stage
also requires adapting the rollout pipeline, which currently computes rewards
and advantages itself before publishing samples.

### Implement and register the service

A new role declares a typed config whose `service_cls` points to an importable
service class. For example, a custom GPU scoring role could start with this
config in `recipe/my_scoring.py`:

```python
from dataclasses import dataclass
from typing import ClassVar

from meshy.config import ServiceConfig


@dataclass
class ScoringServiceConfig(ServiceConfig):
    role: ClassVar[str] = "scoring"
    service_cls: ClassVar[str] = "recipe.my_scoring:ScoringService"
    endpoint_port_base: ClassVar[int] = 33000
    dist_port_base: ClassVar[int] = 43000

    model_path: str
    batch_size: int = 16
    partition_id: str = "data.train"
```

This config is an extension template: implement `ScoringService` before adding
it to a recipe. No registry edit is required; service resolution reads the
config class's `service_cls` attribute.

For a distributed GPU role, subclass
[SpmdService](../meshy/service/spmd.py) and implement these hooks:

1. `from_info(info, my_gpu, topology, runtime)` reads `info.config`, resolves
   replica GPUs and ports from `info`, and supplies constructor arguments.
   Resolve TQ endpoints with
   `meshy.transferqueue.client.resolve_endpoints_file(runtime.root)`.
2. `build_engine()` constructs the engine inside the spawned child process.
   Pass replica rank, world size, device, and distributed address information
   as shown in [TitanTrainingService](../meshy/service/training.py).
3. `build_worker(engine, colocation)` constructs the worker around the
   initialized engine and the service's colocation manager.

The shared template handles process spawning, `engine.init()`, worker thread
startup, command loops, and readiness markers. Keep heavyweight imports and
GPU initialization inside the child-process hooks: the launcher imports the
recipe to inspect placement before starting the run.

For a CPU role, set `uses_gpu: ClassVar[bool] = False` on its config and use
`n_gpus_per_replica=0`. Subclass [Service](../meshy/service/base.py), implement
`from_info()`, `ignite()`, and `wait_for_ready()`, and record child processes
in `self.processes`. Publish readiness with `runtime.mark_ready(self.name)`
once initialization is complete. [RolloutService](../meshy/service/rollout.py)
is the existing CPU-process example.

GPU configs must declare both port bases; derived ports must not collide with
other services on the same host. A colocated custom GPU service also needs
engine acquire/release callbacks that restore and release GPU memory on all
ranks, plus worker calls to request, wait for, and release the GPU token.
See [colocation.md](colocation.md) for that protocol.

## 3. Assemble and Launch the Recipe

### Declare service groups and GPU ownership

Each `ServiceGroup` combines a config with its placement:

| Field | Meaning |
|---|---|
| `id` | Unique group name, referenced by dependencies and colocation rings |
| `config` | Typed service config; its class determines the role and implementation |
| `n_replicas` | Number of independent replicas |
| `n_gpus_per_replica` | Cards per replica; zero for CPU services |
| `colocate_with` | Reuse the GPU block of an earlier group |
| `wait_until` | Group IDs or service names that must be ready before startup |

Declare dependencies before their dependents. Groups without `colocate_with`
claim new cards in declaration order. A colocated group must use the same
total number of cards as its target, but may partition them differently into
replicas. For example, eight TP1 inference replicas can share cards with one
eight-card trainer.

GPU placement and ownership are separate declarations: `colocate_with` reuses
cards, while `COLOCATIONS` schedules access to those cards. Shared GPU services
need both. Configure SGLang with `enable_memory_saver=True` and make the
trainer wait for inference readiness so inference releases memory before the
trainer initializes.

### Complete example

The following module, `recipe/my_grpo.py`, uses one GPU shared by inference and
training. Adjust sequence length and memory settings for your hardware.

```python
from meshy.config import (
    InferenceServiceConfig,
    RolloutServiceConfig,
    TrainerConfig,
    TrainerParamsConfig,
    TrainingServiceConfig,
)
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor

MODEL = "Qwen/Qwen3-1.7B"
PROMPT_BATCH = 4
GROUP_SIZE = 4
TRAIN_BATCH = PROMPT_BATCH * GROUP_SIZE

SERVICE_GROUPS = [
    ServiceGroup(
        id="actor_infer",
        config=InferenceServiceConfig(
            model_path=MODEL,
            server_args={"tp_size": 1, "enable_memory_saver": True},
        ),
        n_replicas=1,
        n_gpus_per_replica=1,
    ),
    ServiceGroup(
        id="actor_train",
        config=TrainingServiceConfig(
            model_path=MODEL,
            trainer_config=TrainerConfig(
                model_name="qwen3",
                model_flavor="1.7B",
                seq_len=2048,
                steps=1000,
                lr=1e-6,
                dp_shard_degree=-1,
                tp_degree=1,
                cp_degree=1,
            ),
            trainer_params=TrainerParamsConfig(
                mini_batch_size=1,
                micro_batch_size=1,
                old_logprobs_source="rollout",
            ),
            batch_size=TRAIN_BATCH,
            weight_sync_mode="disk",
        ),
        n_replicas=1,
        n_gpus_per_replica=1,
        colocate_with="actor_infer",
        wait_until=["actor_infer"],
    ),
    ServiceGroup(
        id="rollout",
        config=RolloutServiceConfig(
            model_path=MODEL,
            dataset="meshy.dataset.gsm8k:GSM8K",
            dataset_kwargs={
                "batch_size": PROMPT_BATCH,
                "split": "train",
                "seed": 42,
            },
            reward="meshy.dataset.gsm8k:GSM8K.reward",
            group_size=GROUP_SIZE,
            sampling_params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": 1024,
            },
            async_max_running_request=TRAIN_BATCH,
            pacing_window=1,
            num_epochs=1,
            filter_zero_std_groups=False,
        ),
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_infer", "actor_train"],
    ),
]

COLOCATIONS = [
    ColocationRing(
        group_id="actor_cards",
        ring=(
            ("actor_infer", SchedulingMode.FALLBACK),
            ("actor_train", SchedulingMode.ON_DEMAND),
        ),
    ),
]


def main() -> None:
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
```

To use the custom dataset from Section 1, replace the rollout's `dataset`,
`dataset_kwargs`, and `reward`. To use separate inference and training GPUs,
remove the trainer's `colocate_with`, set `COLOCATIONS = []`, and optionally
disable SGLang's memory saver. That version of this example requires two GPUs.

`TrainerConfig.model_name` and `model_flavor` must identify a supported Titan
model matching the checkpoint. Keep prompt plus response length within the
configured training sequence length and inference context limit. Trainer
parallel degrees must fit the cards assigned to the training replica.

### Choose pacing and batch sizes

The trainer emits an initial `gen_gate` and another after each weight-sync
boundary. The rollout worker uses `pacing_window` to decide how far generation
may advance:

| `pacing_window` | Behavior |
|---|---|
| `1` or `"auto"` | Lock-step generation, one training batch per gate |
| Integer `N >= 2` | Bounded overlap with an initial budget of `N` training batches |
| `None` | No generation budget after the initial gate; available gates still update version tracking |

Use a training batch size divisible by `group_size`: pacing reserves a whole
response group at once. `async_max_running_request` controls concurrent work
in groups, rounded up to `ceil(limit / group_size)` when positive; use a
multiple of `group_size` for an exact sample limit.

With `stream_minibatch=True`, the trainer consumes chunks of
`mini_batch_size * dp_size` and synchronizes weights after accumulating
`batch_size` samples. Make the training batch divisible by this chunk size.
See [justrl_async.py](../recipe/justrl_async.py) for bounded overlap and
[justrl_fully_async.py](../recipe/justrl_fully_async.py) for streaming training
on separate GPUs.

Leave `filter_zero_std_groups=False` for the initial run. Filtered groups and
failed generations consume pacing budget without producing training samples;
with a finite window, this can prevent the trainer from receiving enough
samples to emit the next gate. Dataset exhaustion can also leave an incomplete
training batch in the queue.

### Launch and inspect the run

Launch the new module from the repository root:

```bash
python scripts/launch.py --recipe recipe.my_grpo
```

The launcher imports `SERVICE_GROUPS`, derives the required physical GPU count,
starts TransferQueue, and runs one torchrun ignitor per card. TQ configuration
is derived automatically; a recipe may export `TRANSFER_QUEUE` to override
it. Use the launcher for an initial run so this infrastructure is started too.

For multiple nodes, run the launcher on every node with matching `--nnodes`,
`--master-addr`, and `--runtime-dir`, and node-specific `--node-rank`. Set
`--nproc-per-node` explicitly to the card count on each node. Dataset/model
paths must be accessible where they are loaded. Weight synchronization
currently uses disk (`"auto"` also resolves to disk), so exported checkpoint
paths must be readable by every inference server that loads them.

By default, artifacts appear under `.xrl_runtime/<timestamp>/`:

- `logs/`: ignitor, service, and TransferQueue process logs.
- `weights/`: exported weights used for synchronization.
- `tensorboard/`: training and rollout metrics.
- `trajectories.jsonl`: generated responses, rewards, advantages, and version
  stamps; `verbose_trajectory_log=True` adds full token and mask details.

Start with small batches and a small dataset. Check that services become ready,
rollouts receive the initial gate, the trainer consumes a batch, and a new
weight version and gate appear. If training stalls, inspect generation errors,
partition/field agreement, and whether enough samples remain to fill a batch.
`num_epochs` limits dataset iteration; the current service stack does not shut
down all long-lived services automatically when rollout finishes. Stop the
launcher with Ctrl+C after the intended run.

For the detailed lifecycle and data flow, see
[architecture_services.md](architecture_services.md).
