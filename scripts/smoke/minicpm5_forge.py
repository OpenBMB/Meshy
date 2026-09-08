"""Single-GPU ForgeEngine smoke test for MiniCPM5-2.6B.

This exercises the same ``build_forge_config -> TitanTrainer -> ForgeEngine``
path used by Meshy, including Hugging Face checkpoint conversion.

Run from the repository root:

    CUDA_VISIBLE_DEVICES=0 python scripts/smoke/minicpm5_forge.py \
        --model-path /user/zhaotianyun/minicpm5-2.6b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/user/zhaotianyun/minicpm5-2.6b"),
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--dump-folder", default="./outputs/minicpm5_forge_smoke")
    return parser.parse_args()


def _configure_single_rank(master_port: int) -> None:
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))


def _assert_embedding_loaded(model, model_path: Path) -> None:
    """Compare a small unpermuted tensor slice with the source safetensor."""
    import torch
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    with index_path.open() as handle:
        weight_map = json.load(handle)["weight_map"]
    shard_path = model_path / weight_map["model.embed_tokens.weight"]
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        expected = handle.get_slice("model.embed_tokens.weight")[0, :16]

    weight = model.tok_embeddings.weight.detach()
    if isinstance(weight, torch.distributed.tensor.DTensor):
        weight = weight.to_local()
    actual = weight[0, :16].float().cpu()
    torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)


def main() -> int:
    args = _parse_args()
    model_path = args.model_path.resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Not a Hugging Face model directory: {model_path}")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be at least 2")

    _configure_single_rank(args.master_port)

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("This ForgeEngine smoke test requires one CUDA GPU")

    from meshy.config import TrainerConfig
    from meshy.backend.titan import TitanTrainer, build_forge_config

    trainer_config = TrainerConfig(
        model_name="minicpm5",
        model_flavor="2.6B",
        seq_len=args.seq_len,
        steps=1,
        dtype="bfloat16",
        local_batch_size=1,
        global_batch_size=1,
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=False,
        dump_folder=args.dump_folder,
        compile_model=False,
    )
    forge_config = build_forge_config(
        trainer_config,
        hf_model_path=str(model_path),
    )

    trainer = None
    try:
        trainer = TitanTrainer(
            forge_config,
            micro_batch_size=1,
            mini_batch_size=1,
            old_logprobs_source="train",
            timer_enabled=False,
        )
        model = trainer.model_parts[0]
        _assert_embedding_loaded(model, model_path)

        tokens = torch.randint(
            0,
            forge_config.model_spec.model.vocab_size,
            (1, args.seq_len),
            dtype=torch.long,
            device=trainer.device,
        )
        positions = torch.arange(
            args.seq_len,
            dtype=torch.long,
            device=trainer.device,
        ).unsqueeze(0)

        trainer.optimizers.zero_grad()
        with trainer.train_context():
            logprobs = trainer._forward_logprobs(
                tokens[:, :-1],
                tokens[:, 1:],
                positions[:, :-1],
            )
            loss = -logprobs.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {loss.item()}")
            loss.backward()

        finite_grad = any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            and parameter.grad.abs().max().item() > 0
            for parameter in model.parameters()
        )
        if not finite_grad:
            raise RuntimeError("No finite non-zero gradient was produced")

        print(
            "PASS: MiniCPM5-2.6B HF weights loaded through ForgeEngine; "
            f"loss={loss.item():.6f}, params={trainer.model_param_count:,}",
            flush=True,
        )
        return 0
    finally:
        del trainer
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
