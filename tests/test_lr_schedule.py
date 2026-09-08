"""The LR schedule is constant after warmup unless a recipe opts into decay.

torchtitan's scheduler default (``decay_ratio=None``) starts a linear decay
to zero right after warmup; run 870776 inherited it unknowingly.
"""

from __future__ import annotations

import pytest
import torch


def _lr_curve(steps: int, **cfg):
    from meshy.backend.titan.config import build_forge_config
    from meshy.config import TrainerConfig

    forge = build_forge_config(TrainerConfig(model_name="qwen3", model_flavor="debugmodel",
                                             steps=steps, lr=1e-6, **cfg))
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-6)
    sched = forge.lr_scheduler.build(optimizers=[opt], training_steps=steps)
    out = []
    for _ in range(70):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_lr_is_constant_by_default():
    lrs = _lr_curve(600)
    assert all(lr == pytest.approx(1e-6) for lr in lrs)


def test_lr_decay_is_opt_in_and_matches_torchtitan_default():
    lrs = _lr_curve(600, lr_decay_ratio=None)
    # the curve run 870776 logged: 9.983e-7 at step 1, 8.85e-7 at step 69
    assert lrs[1] == pytest.approx(9.9833e-7, rel=1e-4)
    assert lrs[69] == pytest.approx(8.85e-7, rel=1e-4)


def test_lr_config_validation():
    from meshy.config import TrainerConfig

    with pytest.raises(ValueError):
        TrainerConfig(lr_decay_ratio=1.5)
    with pytest.raises(ValueError):
        TrainerConfig(lr_decay_type="step")  # type: ignore[arg-type]
