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
    "Add background class to target"
    # bs, c, h, w, d
    y_bg = 1 - y.sum(1, keepdim=True).clamp(0, 1)
    y = torch.cat([y,y_bg], 1)
    y = y / y.sum(1, keepdim=True)
    return y


class Mixup(nn.Module):
    """
    Mixup augmentation for 3D data.
    Args:
        mix_beta (float): Beta distribution parameter for mixup.
        mixadd (bool): If True, mixup is applied to the target as well.
    """
    def __init__(
            self, 
            mix_beta,
            mixadd=False
        ):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Z=None):
        bs = X.shape[0]  # batch size
        perm = torch.randperm(bs) # random permutation of the batch
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        X_coeffs = coeffs.view((-1,) + (1,)*(X.ndim-1))
        Y_coeffs = coeffs.view((-1,) + (1,)*(Y.ndim-1))
        
        X = X_coeffs * X + (1-X_coeffs) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y[perm]
                
        if Z:
            return X, Y, Z

        return X, Y
    