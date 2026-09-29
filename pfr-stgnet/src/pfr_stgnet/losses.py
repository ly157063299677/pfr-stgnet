"""Equation (14): response, paired residual, blind force, blind configuration."""

import torch
from torch.nn import functional as F


TWIST = slice(4, 8)


def _continuous_mask(dim, device):
    mask = torch.ones(dim, dtype=torch.bool, device=device)
    mask[TWIST] = False
    return mask


def objective(output, sample, pair_weight=0.5):
    target = sample["y"]
    mean, logvar = output["mean"], output["logvar"]
    mask = _continuous_mask(target.shape[-1], target.device)
    gaussian = 0.5 * (torch.exp(-logvar[:, mask]) * (mean[:, mask] - target[:, mask]).square()
                      + logvar[:, mask]).mean()
    angular = (1 - torch.cos(mean[:, TWIST] - target[:, TWIST])).mean()
    response = gaussian + angular
    paired = target.new_zeros(())
    if sample["paired"] and sample["force_pn"].item() != 0:
        delta = target - sample["zero_y"]
        predicted = output["residual_mean"]
        paired = (predicted[:, mask] - delta[:, mask]).square().mean()
        paired = paired + (1 - torch.cos(predicted[:, TWIST] - delta[:, TWIST])).mean()
    force = ((output["force_hat"] - sample["force_pn"]) / 250).square()
    configuration = target.new_zeros(())
    if sample["force_pn"].item() != 0:
        configuration = F.cross_entropy(output["configuration_logits"][None],
                                        sample["configuration"][None])
    total = response + pair_weight * paired + 0.2 * force + 0.2 * configuration
    return total, {"response": response.detach(), "pair": paired.detach(),
                   "force": force.detach(), "configuration": configuration.detach()}
