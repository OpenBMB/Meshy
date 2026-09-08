"""Module smoke test: AgentLoopService driver (_amain) over TransferQueue.

Runs the real unified driver against a **real SGLang inference server**, a
**real TransferQueue cluster**, and a scripted fake Training Service (separate
process) that emits gen gates and consumes GRPO batches from the queue.
Verifies the lock-step (window=1) control loop end to end: gate_0 releases the
first rollout window, samples land in the data partition with the full column
contract, the fake trainer clears them and raises gate_1, and the driver
finishes once the (tiny) dataset is exhausted.

    HF_ENDPOINT=https://hf-mirror.com XRL_RUNTIME_DIR=/tmp/xrl_rt_al \
    CUDA_VISIBLE_DEVICES=0 XRL_MODEL=<local qwen3-0.6B path> \
    python scripts/smoke/agentloop.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from meshy.service.rollout import _amain
from meshy.service.base import GPU
from meshy.service.inference import SGLangService
from meshy.service.runtime import RuntimeDir
from meshy.transferqueue.client import resolve_endpoints_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROLLOUT_BATCH = 2
GROUP_SIZE = 2
TRAIN_BATCH = ROLLOUT_BATCH * GROUP_SIZE
ROUNDS = 2  # dataset: train[:4] at batch 2 -> two rollout batches

_FAKE_TRAINER = f"""
import json, sys, time
sys.path.insert(0, {REPO_ROOT!r})
from meshy.transferqueue import adapter, connect, control
from meshy.config import GRPO_TRAINER_FIELDS

EP = sys.argv[1]
TRAIN_BATCH, ROUNDS = {TRAIN_BATCH}, {ROUNDS}
client = connect(EP)
control.emit_gen_gate(client, step=0, weight_version=0)

rounds = []
deadline = time.monotonic() + 600
for r in range(ROUNDS):
    while True:
        assert time.monotonic() < deadline, f"round {{r}}: batch never arrived"
        meta = client.get_meta(
            data_fields=GRPO_TRAINER_FIELDS, batch_size=TRAIN_BATCH,
            partition_id="data.train", mode="fetch", task_name="trainer",
        )
        if meta.size == TRAIN_BATCH:
            break
        time.sleep(0.2)
    samples = adapter.td_to_samples(client.get_data(meta), GRPO_TRAINER_FIELDS)
    rounds.append({{"versions": sorted(int(s["weight_version"]) for s in samples)}})
    client.clear_samples(meta)
    control.emit_gen_gate(client, step=r + 1, weight_version=r + 1)
print(json.dumps({{"rounds": rounds}}))
"""


def main() -> int:
    from meshy.transferqueue.launch import TransferQueueCluster

    model = os.environ["XRL_MODEL"]
    runtime = RuntimeDir(os.environ.get("XRL_RUNTIME_DIR", "/tmp/xrl_rt_al"))
    endpoints_file = resolve_endpoints_file(runtime.root)

    gpu = GPU(host="127.0.0.1", global_rank=0, node_rank=0, local_rank=0)
    infer = SGLangService(
        name="inference-0",
        my_gpu=gpu,
        replica_gpus=[gpu],
        server_args={"model_path": model, "tp_size": 1, "mem_fraction_static": 0.5},
        endpoint_port=30000,
        dist_port=40000,
        is_colocate=False,
        runtime=runtime,
    )

    kwargs = {
        "model_path": model,
        "dataset": "meshy.dataset.gsm8k:GSM8K",
        "dataset_kwargs": {"batch_size": ROLLOUT_BATCH, "split": "train[:4]", "seed": 42},
        "reward": "meshy.dataset.gsm8k:GSM8K.reward",
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 64},
        "group_size": GROUP_SIZE,
        "num_epochs": 1,
        "async_max_running_request": -1,
        "pacing_window": 1,
        "poll_interval": 1.0,
        "train_batch_size": TRAIN_BATCH,
        "partition_id": "data.train",
        "tq_endpoints_file": endpoints_file,
        # Endpoints are injected by the ignitor from the derived topology; the
        # smoke test supplies the single real inference endpoint directly.
        "inference_endpoints": None,  # filled once the server is up
    }

    ok = False
    with TransferQueueCluster(
        endpoints_file,
        num_storage_units=2,
        storage_unit_size=1024,
        pre_alloc_sample_num=4 * TRAIN_BATCH,
        log_dir=runtime.root,
    ):
        trainer = None
        try:
            infer.ignite()
            infer.wait_for_ready()
            print(f"[agentloop] inference ready at {infer.endpoint}", flush=True)
            kwargs["inference_endpoints"] = [infer.endpoint]

            trainer = subprocess.Popen(
                [sys.executable, "-c", textwrap.dedent(_FAKE_TRAINER), endpoints_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            t0 = time.monotonic()
            asyncio.run(asyncio.wait_for(_amain(kwargs, runtime), timeout=900))
            out, err = trainer.communicate(timeout=120)
            print(f"[agentloop] driver finished in {time.monotonic() - t0:.1f}s", flush=True)
            if trainer.returncode != 0:
                print(f"[agentloop] fake trainer failed:\n{out}\n{err}", flush=True)
            else:
                result = json.loads(
                    [line for line in out.splitlines() if line.startswith("{")][-1]
                )
                print(f"[agentloop] fake trainer rounds = {result['rounds']}", flush=True)
                ok = (
                    len(result["rounds"]) == ROUNDS
                    and result["rounds"][0]["versions"] == [0] * TRAIN_BATCH
                    and result["rounds"][1]["versions"] == [1] * TRAIN_BATCH
                )
        finally:
            if trainer is not None and trainer.poll() is None:
                trainer.kill()
            infer.terminate()
            for p in infer.processes:
                p.join(timeout=20)
    print("[agentloop] RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
