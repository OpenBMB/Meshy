"""VAPO normalization and clipped value objective."""
import torch
import torch.distributed as dist


def whiten_advantages(samples, seq_len, device, mesh=None, replication=1):
    from .batch import ADVANTAGE_TOKENS

    stats = torch.zeros(3, dtype=torch.float64, device=device)
    entries = []
    for sample in samples:
        advantage = sample[ADVANTAGE_TOKENS]
        length = min(len(advantage), seq_len)
        mask = torch.zeros_like(advantage, dtype=torch.bool)
        if length > 1:
            mask[:length - 1] = sample['mask_assistant'][1:length].bool()
        selected = advantage[mask].double()
        stats += torch.stack((selected.sum(), selected.square().sum(),
                              selected.new_tensor(selected.numel()))).to(device)
        entries.append((advantage, mask))
    if mesh is not None:
        # Full sequences are replicated on CP ranks at this stage.
        stats /= replication
        dist.all_reduce(stats, group=mesh.get_group())
    total, squares, count = stats
    mean = total / count.clamp(min=1)
    variance = (squares / count.clamp(min=1) - mean.square()).clamp(min=0)
    if count >= 2:
        variance *= count / (count - 1)
    for advantage, mask in entries:
        advantage.copy_(torch.where(mask, (advantage - mean.to(advantage)) *
                                   torch.rsqrt(variance.to(advantage) + 1e-8), 0))


def clipped_value_error(values, targets, old_values, clip):
    error = (values.float() - targets.float()).square()
    if clip is None:
        return error
    old = old_values.detach().float()
    clipped = old + (values.float() - old).clamp(-clip, clip)
    return torch.maximum(error, (clipped - targets.float()).square())


def critic_parameter_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim <= 1 or name.endswith('.bias') else decay).append(parameter)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]
