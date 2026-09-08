"""Module smoke test: TitanTrainingService (standalone, 1 card, TQ-fed).

Starts a real TransferQueue cluster, spawns the trainer engine (torchtitan
env), and plays the AgentLoop's part by hand: wait for the trainer's gate_0
pulse, put one ``batch_size`` GRPO batch (full column contract incl.
``weight_version``) into the data partition, then wait for gate_1 -- which the
trainer only raises after training the batch, dumping the HF checkpoint and
bumping its version. Verifies /health reports version 1 and the checkpoint
exists.

    HF_ENDPOINT=https://hf-mirror.com XRL_RUNTIME_DIR=/tmp/xrl_rt_train \
    CUDA_VISIBLE_DEVICES=1 XRL_MODEL=<local qwen3-0.6B path> \
    python scripts/smoke/train.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import requests

from meshy.config import GRPO_TRAINER_FIELDS, TrainerConfig, TrainerParamsConfig
from meshy.service.base import GPU
from meshy.service.runtime import RuntimeDir
from meshy.service.training import TitanTrainingService
from meshy.transferqueue.client import resolve_endpoints_file

SEQ_LEN = int(os.environ.get("XRL_SEQ_LEN", "512"))
BATCH_SIZE = int(os.environ.get("XRL_BATCH", "2"))
PARTITION = "data.train"


def _make_samples(n: int, length: int = 24):
    import torch
    from tensordict import TensorDict

    out = []
    for _ in range(n):
        out.append(
            TensorDict(
                {
                    "tokens": torch.randint(1, 1000, (length,), dtype=torch.long),
                    "logprobs": torch.randn(length, dtype=torch.float32) * 0.1 - 0.5,
                    "mask_assistant": torch.ones(length, dtype=torch.float32),
                    "advantage": torch.tensor(float(torch.randn(())), dtype=torch.float32),
                    "weight_version": torch.tensor(0, dtype=torch.int64),
                    "reward": torch.tensor(1.0, dtype=torch.float32),
                    "truncated": torch.tensor(0, dtype=torch.int64),
                    "repetition": torch.tensor(0, dtype=torch.int64),
                    "mixed_version": torch.tensor(0, dtype=torch.int64),
                },
                batch_size=[],
            )
        )
    return out


def main() -> int:
    from meshy.transferqueue import adapter, connect
    from meshy.transferqueue import control
    from meshy.transferqueue.launch import TransferQueueCluster

    model = os.environ["XRL_MODEL"]
    runtime = RuntimeDir(os.environ.get("XRL_RUNTIME_DIR", "/tmp/xrl_rt_train"))
    endpoints_file = resolve_endpoints_file(runtime.root)

    trainer_config = TrainerConfig(
        model_name=os.environ.get("XRL_MODEL_NAME", "qwen3"),
        model_flavor=os.environ.get("XRL_MODEL_FLAVOR", "0.6B"),
        seq_len=SEQ_LEN,
        steps=1000,
        dtype="bfloat16",
        lr=1e-6,
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=False,
        dump_folder="./outputs/smoke_train",
        compile_model=False,
    )
    trainer_params = TrainerParamsConfig(
        mini_batch_size=1,
        micro_batch_size=1,
        ppo_clip_eps_low=0.2,
        ppo_clip_eps_high=0.2,
        old_logprobs_source="rollout",
    )

    gpu = GPU(host="127.0.0.1", global_rank=0, node_rank=0, local_rank=0)
    ok = False
    with TransferQueueCluster(
        endpoints_file,
        num_storage_units=2,
        storage_unit_size=1024,
        pre_alloc_sample_num=max(4 * BATCH_SIZE, 16),
        log_dir=runtime.root,
    ):
        service = TitanTrainingService(
            name="training-0",
            my_gpu=gpu,
            replica_gpus=[gpu],
            endpoint_port=31000,
            dist_port=41000,
            is_colocate=False,
            model_path=model,
            trainer_config=trainer_config,
            trainer_params=trainer_params,
            batch_size=BATCH_SIZE,
            timer_enabled=False,
            tq_endpoints_file=endpoints_file,
            partition_id=PARTITION,
            tq_fields=list(GRPO_TRAINER_FIELDS),
            runtime=runtime,
        )
        try:
            t0 = time.monotonic()
            service.ignite()
            service.wait_for_ready()
            print(
                f"[train] READY in {time.monotonic() - t0:.1f}s endpoint={service.endpoint}",
                flush=True,
            )

            client = connect(endpoints_file)
            v0 = control.wait_gen_gate(client, 0, timeout=300)
            print(f"[train] gate_0 up (weight_version={v0})", flush=True)

            td = adapter.samples_to_td(_make_samples(BATCH_SIZE), GRPO_TRAINER_FIELDS)
            client.put(data=td, partition_id=PARTITION)
            print(f"[train] put {BATCH_SIZE} samples into {PARTITION!r}", flush=True)

            t1 = time.monotonic()
            v1 = control.wait_gen_gate(client, 1, timeout=1800)
            print(
                f"[train] gate_1 up (weight_version={v1}) after {time.monotonic() - t1:.1f}s",
                flush=True,
            )

            health = requests.get(f"{service.endpoint}/health", timeout=10).json()
            print(f"[train] health = {health}", flush=True)

            weights_dir = os.path.join(runtime.root, "weights", "training-0", "v1")
            has_weights = os.path.isdir(weights_dir) and any(
                f.endswith(".safetensors") for f in os.listdir(weights_dir)
            )
            print(f"[train] checkpoint at {weights_dir}: exists={has_weights}", flush=True)

            ok = v0 == 0 and v1 == 1 and health.get("version") == 1 and has_weights
        finally:
            service.terminate()
            for p in service.processes:
                p.join(timeout=30)
    print("[train] RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
