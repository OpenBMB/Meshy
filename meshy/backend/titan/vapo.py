"""Clipped value objective and critic optimizer groups."""
import torch


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
