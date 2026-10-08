"""Exercise real DCP save/load with legacy and current optimizer layouts."""
import copy
import tempfile
import unittest

import torch
import torch.distributed.checkpoint as dcp

from meshy.backend.titan.critic.engine import CriticEngine
from meshy.backend.titan.vapo import critic_parameter_groups


def make_engine(grouped, explicit=True):
    engine = CriticEngine.__new__(CriticEngine)
    engine.model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    params = critic_parameter_groups(engine.model, .1) if grouped else engine.model.parameters()
    engine.optimizer = torch.optim.AdamW(
        params, lr=5e-6, betas=(.9, .98) if grouped else (.9, .999),
        eps=1e-8 if grouped else 1e-7, weight_decay=.1 if grouped else .01,
    )
    engine.restore_optimizer_config = explicit
    engine.train_steps = 4
    engine.base_lr = 5e-6
    engine.warmup_steps = 10
    return engine


def step(engine):
    engine.optimizer.zero_grad()
    engine.model(torch.tensor([[1., 2., 3.], [2., -1., 4.]])).square().sum().backward()
    engine.optimizer.step()


class CriticCheckpointMigration(unittest.TestCase):
    def check_restore(self, grouped):
        source = make_engine(grouped)
        for _ in range(4):
            step(source)
        with tempfile.TemporaryDirectory(prefix='meshy-critic-migration-') as directory:
            source.save_state(directory)
            restored = make_engine(True)
            restored.load_state(directory)
        self.assertEqual(len(restored.optimizer.param_groups), 2)
        for group, decay in zip(restored.optimizer.param_groups, [.1, 0.]):
            self.assertEqual(group['betas'], (.9, .98))
            self.assertEqual(group['weight_decay'], decay)
            self.assertEqual(group['eps'], 1e-8)
        # The migration must preserve each parameter's moments and step,
        # even though parameter order across optimizer groups has changed.
        expected = make_engine(True)
        expected.model.load_state_dict(source.model.state_dict())
        for old, new, ref in zip(source.model.parameters(), restored.model.parameters(), expected.model.parameters()):
            torch.testing.assert_close(old, new)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(source.optimizer.state[old][key], restored.optimizer.state[new][key])
            expected.optimizer.state[ref] = copy.deepcopy(source.optimizer.state[old])
        # SpmdEngine restores train_steps from the checkpoint manifest.
        # Verify that the next warmup update uses that position and base LR.
        for engine in (restored, expected):
            engine.train_steps = source.train_steps + 1
            engine._apply_warmup()
            self.assertEqual(engine.optimizer.param_groups[0]['lr'], 2.5e-6)
            step(engine)
        for actual, reference in zip(restored.model.parameters(), expected.model.parameters()):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_legacy_single_group_to_current_groups(self):
        self.check_restore(False)

    def test_current_checkpoint_roundtrip(self):
        self.check_restore(True)

    def test_implicit_config_preserves_legacy_resume_behavior(self):
        source = make_engine(False)
        step(source)
        with tempfile.TemporaryDirectory(prefix='meshy-critic-migration-') as directory:
            source.save_state(directory)
            restored = make_engine(False, explicit=False)
            restored.optimizer.param_groups[0]['betas'] = (.9, .98)
            restored.load_state(directory)
        self.assertEqual(restored.optimizer.param_groups[0]['betas'], (.9, .999))

    def test_missing_moments_still_fail_strictly(self):
        source = make_engine(False)
        step(source)
        state = source._dcp_state()
        first = next(iter(state['optim']['state'].values()))
        del first['exp_avg_sq']
        with tempfile.TemporaryDirectory(prefix='meshy-critic-migration-') as directory:
            dcp.save(state, checkpoint_id=directory)
            with self.assertRaises(dcp.CheckpointException):
                make_engine(True).load_state(directory)


if __name__ == '__main__':
    unittest.main()
