"""CPU regression tests; runnable with unittest without extra dependencies."""
import contextlib
import os
import tempfile
import unittest
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from meshy.backend.titan.batch import ADVANTAGE_TOKENS
from meshy.backend.titan.vapo import whiten_advantages, clipped_value_error, critic_parameter_groups


def row(values):
    return {ADVANTAGE_TOKENS: torch.tensor(values, dtype=torch.float32),
            'mask_assistant': torch.tensor([0] + [1] * (len(values) - 1))}


def distributed_whiten(rank, rendezvous):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4)
    try:
        # Two DP slices, each replicated on two CP ranks.
        sample = row([1., 2., 99.] if rank < 2 else [4., 8., 99.])
        whiten_advantages([sample], 3, 'cpu', NS(get_group=lambda: dist.group.WORLD), 2)
        all_values = torch.tensor([1., 2., 4., 8.])
        expected = (all_values - all_values.mean()) / (all_values.var() + 1e-8).sqrt()
        torch.testing.assert_close(sample[ADVANTAGE_TOKENS][:2], expected[:2] if rank < 2 else expected[2:])
        assert sample[ADVANTAGE_TOKENS][-1] == 0
    finally:
        dist.destroy_process_group()


class VapoAlignment(unittest.TestCase):
    def test_whitening_masks_and_constant(self):
        sample = row([2., 4., 1000.])
        whiten_advantages([sample], 3, 'cpu')
        torch.testing.assert_close(sample[ADVANTAGE_TOKENS], torch.tensor([-2**-.5, 2**-.5, 0.]))
        sample = row([2., 2., 1000.])
        whiten_advantages([sample], 3, 'cpu')
        self.assertEqual(sample[ADVANTAGE_TOKENS].abs().sum(), 0)
        whiten_advantages([], 3, 'cpu')

    def test_distributed_whitening_counts_unique_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_whiten, args=(os.path.join(directory, 'rendezvous'),), nprocs=4)

    def test_value_clipping_gradient_and_old_snapshot(self):
        old = torch.tensor([0., 0.], requires_grad=True)
        values = torch.tensor([0.5, -0.5], requires_grad=True)
        loss = clipped_value_error(values, torch.ones(2), old, .2).sum()
        torch.testing.assert_close(loss, torch.tensor(.64 + 2.25))
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([0., -3.]))
        self.assertIsNone(old.grad)

    def test_optimizer_groups(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
        groups = critic_parameter_groups(model, .1)
        self.assertEqual(len(groups[0]['params']), 1)
        self.assertEqual(len(groups[1]['params']), 3)
        optimizer = torch.optim.AdamW(groups, lr=5e-6, betas=(.9, .98), eps=1e-8)
        self.assertEqual(optimizer.param_groups[0]['betas'], (.9, .98))
        self.assertEqual(optimizer.param_groups[1]['weight_decay'], 0)

    def test_critic_accumulation_freezes_values_across_epochs(self):
        from meshy.backend.titan.critic.engine import CriticEngine
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.))
            def forward(self, inputs, **kwargs):
                return inputs.float().unsqueeze(-1) * self.weight
        model = Model()
        stub = NS(device=torch.device('cpu'), train_steps=0, value_clip=.2,
                  _apply_warmup=lambda: None, _global_sum=lambda x: x,
                  sharder=NS(shard_seq=lambda *xs: xs, gather_seq=lambda *xs: xs),
                  train_context=contextlib.nullcontext, model=model,
                  clip_grad_norm=lambda: None, optimizer=torch.optim.SGD(model.parameters(), lr=.25))
        batch = (torch.ones(1, 2, dtype=torch.long), torch.zeros(1, 2),
                 torch.ones(1, 2), torch.ones(1, 2), None)
        first, old = CriticEngine.train_value_accumulated(stub, [batch, batch], collect_values=True)
        self.assertAlmostEqual(first, 1.)
        self.assertAlmostEqual(model.weight.item(), .5)
        self.assertAlmostEqual(old[0].abs().sum().item(), 0.)
        second, _ = CriticEngine.train_value_accumulated(stub, [batch, batch], old_values=old)
        self.assertAlmostEqual(second, .64, places=6)
        self.assertAlmostEqual(model.weight.item(), .5)
        self.assertEqual(old[0].abs().sum().item(), 0.)

    def test_icepop_masks_off_policy_tokens(self):
        from meshy.backend.titan.batch import Batch
        from meshy.backend.titan.plan import MiniPlan
        from meshy.backend.titan.trainer import TitanTrainer
        stub = NS(ppo_clip_eps_low=.2, ppo_clip_eps_high=.28, use_tis=True, use_icepop=True,
                  tis_ratio_min=.5, tis_ratio_max=5., calculate_per_token_loss=True,
                  old_logprobs_source='train',
                  sharder=NS(all_reduce_sum=lambda x: x), parallel_dims=NS(cp=1))
        old = torch.full((1, 3), -2.)
        new = old.clone().requires_grad_()
        batch = Batch(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long),
                      torch.zeros(1, 3, dtype=torch.int32), torch.ones(1, 3),
                      torch.zeros(1, 3, dtype=torch.long), torch.ones(1),
                      old - torch.tensor([[2., 6., .25]]).log(), torch.tensor([3]), 1)
        mini = MiniPlan(micros=(), n_docs_global=1, n_tokens_global=3, tokens_per_rank=(3,))
        loss, sums = TitanTrainer._ppo_clip_loss(stub, new, old, batch, mini)
        self.assertAlmostEqual(float(loss.detach()), -2 / 3, places=6)
        self.assertEqual(float(sums['tis_masked_sum']), 2.)
        self.assertEqual(float(sums['ratio_sum']), 3.)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.tensor([[-2. / 3, 0., 0.]]))

        stub.calculate_per_token_loss = False
        batch.rollout_logprobs = old - torch.tensor([[.5, 5., 1.]]).log()
        loss, sums = TitanTrainer._ppo_clip_loss(stub, old.clone().requires_grad_(), old, batch, mini)
        self.assertEqual(float(sums['tis_masked_sum']), 0.)
        self.assertAlmostEqual(float(loss.detach()), -6.5 / 3, places=6)

        stub.use_icepop = False
        loss, sums = TitanTrainer._ppo_clip_loss(stub, old, old, batch, mini)
        self.assertAlmostEqual(float(loss), -1.)

    def test_metric_tags_separate_ois_and_tis(self):
        from meshy.backend.titan.metrics import TitanTrainer
        metrics = TitanTrainer._training_metrics(NS(), {
            'ratio_mean': 1.1, 'tis_ratio_mean': 1.4, 'pg_loss': -.2,
        })
        self.assertEqual(metrics['train/ois'], 1.1)
        self.assertEqual(metrics['train/tis'], 1.4)
        self.assertEqual(metrics['train/pg_loss'], -.2)
        self.assertEqual(metrics['train/loss'], -.2)

    def test_window_snapshot_precedes_all_updates(self):
        from unittest.mock import patch
        from meshy.backend.titan.trainer import TitanTrainer
        events = []
        plans = [NS(micros=()), NS(micros=())]
        def snapshot(samples, mini):
            events.append('snapshot')
            return [torch.tensor(0.)]
        def update(samples, mini, timer, old):
            events.append('update')
            self.assertEqual(float(old[0]), 0.)
            return {}, None
        stub = NS(timer_enabled=False, enable_gae=False, old_logprobs_source='train',
                  use_icepop=True,
                  _compute_old_logprobs_train=snapshot, _run_mini_batch=update,
                  lr_schedulers=NS(step=lambda: None), step=0, batch_layout='padded',
                  _aggregate_metrics=lambda *args: {})
        with patch('meshy.backend.titan.trainer.plan_stats', return_value={}):
            TitanTrainer.train_step(stub, [], plan=plans)
        self.assertEqual(events, ['snapshot', 'snapshot', 'update', 'update'])


if __name__ == '__main__':
    unittest.main()
