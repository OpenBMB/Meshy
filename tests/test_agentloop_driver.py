"""End-to-end check of the unified AgentLoop driver over a real TransferQueue.

The driver runs in-process (with a fake tokenizer and a fake inference engine),
while a *real* deployed TQ (controller + storage subprocesses) and a scripted
fake Training Service (separate process: emits gen gates, consumes batches,
clears them) sit on the other side. This exercises the whole implicit control
loop the cutover introduced, with no GPU:

* gate_0 releases the first window; the pacer blocks the second batch until the
  fake trainer has consumed round 1 and raised gate_1 (lock-step, window=1);
* samples arrive with the full GRPO column contract and the right
  ``weight_version`` staleness tag per round (round 1 -> v0, round 2 -> v1);
* the driver terminates cleanly once the dataset is exhausted.

Run with the torchtitan env:

    TRANSFER_QUEUE_SRC=/workspace/TransferQueue \\
    python -m pytest tests/test_agentloop_driver.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROLLOUT_BATCH = 2  # prompts per dataset batch
GROUP_SIZE = 2
TRAIN_BATCH = ROLLOUT_BATCH * GROUP_SIZE  # samples per gate window
ROUNDS = 2

# ── stub dataset / reward, importable by dotted path ────────────────────────
_STUB_MODULE = "xrl_test_agentloop_stub"


class StubDataset:
    def __init__(self, batch_size: int, n_batches: int):
        self.batch_size = batch_size
        self.remaining = n_batches

    def next_batch(self, builder):
        if self.remaining <= 0:
            return []
        self.remaining -= 1
        return [
            builder.build_sample([{"role": "user", "content": f"q{self.remaining}-{i}"}])
            for i in range(self.batch_size)
        ]


def stub_reward(sample) -> float:
    return 1.0


def _install_stub_module() -> None:
    mod = types.ModuleType(_STUB_MODULE)
    mod.StubDataset = StubDataset
    mod.reward = stub_reward
    sys.modules[_STUB_MODULE] = mod


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kw):
        return {"input_ids": list(range(100, 100 + 3 * len(messages)))}

    def decode(self, tokens, **kw):
        return "decoded"


class _FakeInferenceEngine:
    """Instant generations; interface-compatible with SGLangEngine."""

    def __init__(self, endpoints, sampling_params=None, **kw):
        self.sampling_params = sampling_params or {}

    async def generate(self, input_ids, sampling_params=None, attempts=None):
        return [7, 8, 9], [-0.1, -0.2, -0.3]


# ── scripted fake Training Service (separate process, real ZMQ) ─────────────
_FAKE_TRAINER = f"""
import json, sys, time
sys.path.insert(0, {REPO_ROOT!r})
from meshy.transferqueue import adapter, connect, control
from meshy.config import GRPO_TRAINER_FIELDS

EP = sys.argv[1]
TRAIN_BATCH, ROUNDS = {TRAIN_BATCH}, {ROUNDS}
client = connect(EP)

# gate_0: release the first rollout window against the base weights (v0).
control.emit_gen_gate(client, step=0, weight_version=0)

rounds = []
deadline = time.monotonic() + 120
for r in range(ROUNDS):
    while True:
        assert time.monotonic() < deadline, f"round {{r}}: batch never arrived"
        meta = client.get_meta(
            data_fields=GRPO_TRAINER_FIELDS, batch_size=TRAIN_BATCH,
            partition_id="data.train", mode="fetch", task_name="trainer",
        )
        if meta.size == TRAIN_BATCH:
            break
        time.sleep(0.1)
    samples = adapter.td_to_samples(client.get_data(meta), GRPO_TRAINER_FIELDS)
    rounds.append({{
        "versions": sorted(int(s["weight_version"]) for s in samples),
        "advantages": [float(s["advantage"]) for s in samples],
        "lengths": sorted(int(s["tokens"].numel()) for s in samples),
    }})
    client.clear_samples(meta)
    # "train + weight sync" done -> raise the next gate.
    control.emit_gen_gate(client, step=r + 1, weight_version=r + 1)

print(json.dumps({{"rounds": rounds}}))
"""


def test_unified_driver_lockstep_roundtrip(tmp_path, monkeypatch):
    from meshy.transferqueue.launch import TransferQueueCluster

    _install_stub_module()

    import meshy.engine.sglang as inf_mod
    import meshy.utils.sample as sample_mod

    monkeypatch.setattr(inf_mod, "SGLangEngine", _FakeInferenceEngine)
    monkeypatch.setattr(
        sample_mod,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeTokenizer()),
    )

    from meshy.service.rollout import _amain
    from meshy.service.runtime import RuntimeDir
    from meshy.transferqueue.client import store_ref

    # Deployed default: endpoint discovery via the bootstrap store, no files.
    endpoints = store_ref(str(tmp_path))
    runtime = RuntimeDir(str(tmp_path / "runtime"))
    kwargs = {
        "model_path": "stub-model",
        "dataset": f"{_STUB_MODULE}:StubDataset",
        "dataset_kwargs": {"batch_size": ROLLOUT_BATCH, "n_batches": ROUNDS},
        "reward": f"{_STUB_MODULE}:reward",
        "sampling_params": {"max_new_tokens": 8},
        "group_size": GROUP_SIZE,
        "num_epochs": 1,
        "async_max_running_request": -1,
        "pacing_window": 1,
        "poll_interval": 0.2,
        "train_batch_size": TRAIN_BATCH,
        "partition_id": "data.train",
        "tq_endpoints_file": endpoints,
        "inference_endpoints": ["http://fake:1"],
        "verbose_trajectory_log": False,
    }

    cluster = TransferQueueCluster(
        endpoints,
        num_storage_units=2,
        storage_unit_size=256,
        pre_alloc_sample_num=2 * TRAIN_BATCH,
        log_dir=str(tmp_path),
    )
    with cluster:
        trainer = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(_FAKE_TRAINER), endpoints],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            asyncio.run(asyncio.wait_for(_amain(kwargs, runtime), timeout=120))
            out, err = trainer.communicate(timeout=60)
        finally:
            if trainer.poll() is None:
                trainer.kill()

    assert trainer.returncode == 0, f"fake trainer failed:\n{out}\n{err}"
    result = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])
    rounds = result["rounds"]
    assert len(rounds) == ROUNDS

    # Round N was generated against the weights gate_N released.
    assert rounds[0]["versions"] == [0] * TRAIN_BATCH
    assert rounds[1]["versions"] == [1] * TRAIN_BATCH
    # GRPO ran: per-group advantage normalization means advantages sum to ~0
    # per group of equal rewards -> all zeros here (identical rewards).
    for r in rounds:
        assert all(abs(a) < 1e-6 for a in r["advantages"]), r["advantages"]
    # Trajectories were logged once per sample.
    traj = (tmp_path / "runtime" / "trajectories.jsonl").read_text().strip().splitlines()
    assert len(traj) == ROUNDS * TRAIN_BATCH
    records = [json.loads(line) for line in traj]
    assert {record["round"] for record in records} == {1, 2}
    assert all(record["weight_version"] == record["round"] - 1 for record in records)
