"""Module smoke test: SGLangService (standalone, 1 card).

Run under the *torchtitan* interpreter (it only launches the sglang child):

    HF_ENDPOINT=https://hf-mirror.com XRL_RUNTIME_DIR=/tmp/xrl_rt_infer \
    CUDA_VISIBLE_DEVICES=0 XRL_MODEL=Qwen/Qwen3-0.6B \
    python scripts/smoke/infer.py

Set COLO=1 to exercise the colocate path (enable_memory_saver + release on ready).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import requests


def _gpu_used_mb(phys_id: str) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={phys_id}"]
    )
    return int(out.decode().strip().splitlines()[0])

from meshy.service.base import GPU
from meshy.service.runtime import RuntimeDir
from meshy.service.inference import SGLangService


def main() -> int:
    model = os.environ.get("XRL_MODEL", "Qwen/Qwen3-0.6B")
    colo = os.environ.get("COLO", "0") == "1"
    runtime = RuntimeDir(os.environ.get("XRL_RUNTIME_DIR", "/tmp/xrl_rt_infer"))

    gpu = GPU(host="127.0.0.1", global_rank=0, node_rank=0, local_rank=0)
    server_args = {"model_path": model, "tp_size": 1, "mem_fraction_static": 0.6}
    if colo:
        server_args["enable_memory_saver"] = True
    worker = SGLangService(
        name="inference-0",
        my_gpu=gpu,
        replica_gpus=[gpu],
        server_args=server_args,
        endpoint_port=30000,
        dist_port=40000,
        is_colocate=colo,
        runtime=runtime,
    )

    ok = False
    try:
        t0 = time.monotonic()
        worker.ignite()
        worker.wait_for_ready()
        print(f"[infer] READY in {time.monotonic() - t0:.1f}s endpoint={worker.endpoint}", flush=True)

        if not colo:
            r = requests.post(
                f"{worker.endpoint}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
                },
                timeout=120,
            )
            r.raise_for_status()
            print(f"[infer] GENERATE -> {r.json().get('text')!r}", flush=True)
        else:
            # colocate: memory was released on ready; measure, resume, re-measure.
            phys = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") or ["0"])[0]
            released_mb = _gpu_used_mb(phys)
            requests.post(
                f"{worker.endpoint}/resume_memory_occupation",
                json={"tags": ["weights", "kv_cache"]},
                timeout=600,
            ).raise_for_status()
            time.sleep(3)
            resumed_mb = _gpu_used_mb(phys)
            print(
                f"[infer] colocate mem: released={released_mb}MiB resumed={resumed_mb}MiB "
                f"(delta={resumed_mb - released_mb}MiB)",
                flush=True,
            )
            assert resumed_mb > released_mb + 200, "resume did not re-occupy GPU memory"
            print("[infer] release/resume verified", flush=True)

        assert runtime.is_ready("inference-0"), "readiness marker not published"
        print("[infer] readiness marker published", flush=True)
        ok = True
    finally:
        worker.terminate()
        for p in worker.processes:
            p.join(timeout=20)
    print("[infer] RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
