"""Compute group difficulty before asynchronous consumers split a group."""


def assign_positive_lm_weights(group):
    rewards = [float(s.reward if s.raw_reward is None else s.raw_reward) for s in group]
    binary = bool(rewards) and all(r in (0.0, 1.0) for r in rewards)
    difficulty = 1.0 - sum(rewards) / len(rewards) if binary else 0.0
    for sample, reward in zip(group, rewards):
        sample.positive_lm_weight = difficulty if binary and reward == 1.0 else 0.0
