import torch
from torch import nn
from torch.distributions import Beta


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def to_ce_target(y):
    "Add the background class to ground truth target channel"
    # y is ground truth labels, [B, C, D, H, W]
    y_bg = 1 - y.sum(1, keepdim=True).clamp(0, 1) # calculate background probability
    y = torch.cat([y,y_bg], 1)  # append background class to the last channel in the target, [B, C+1, D, H, W]
    y = y / y.sum(1, keepdim=True)
    return y


