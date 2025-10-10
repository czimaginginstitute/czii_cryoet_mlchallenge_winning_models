import monai
import torch
from torch import nn
from torch.distributions import Beta

def get_basic_transform_list(keys: list=["input"]):
    """
    Get the augmentation transforms for training and validation.
    Args:
        cfg: Configuration object containing augmentation parameters.
        mode: 'train' or 'val'.
    Returns:
        train_aug: Augmentation transforms for training.
        val_aug: Augmentation transforms for validation.
    """
    transorms = [
        monai.transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        monai.transforms.NormalizeIntensityd(keys=keys)
    ]
    return transorms

train_aug = monai.transforms.Compose([
    *get_basic_transform_list(),
    monai.transforms.RandSpatialCropSamplesd(
        keys=["input", "target"],
        roi_size=(96, 96, 96),
        num_samples=4
    ),
    monai.transforms.RandFlipd(
        keys=["input", "target"],
        prob=0.5,
        spatial_axis=0,
    ),
    monai.transforms.RandFlipd(
        keys=["input", "target"],
        prob=0.5,
        spatial_axis=1,
    ),
    monai.transforms.RandFlipd(
        keys=["input", "target"],
        prob=0.5,
        spatial_axis=2,
    ),
    monai.transforms.RandRotate90d(
        keys=["input", "target"],
        prob=0.75,
        max_k=3,
        spatial_axes=(0, 1),
    ),
    monai.transforms.RandRotated(
        keys=["input", "target"], 
        prob=0.5,range_x=0.78,
        range_y=0.,range_z=0., 
        padding_mode='reflection'
    )
    ])

# return 98 non-overlapping grid pathes per tomogram
val_aug = monai.transforms.Compose([
    *get_basic_transform_list(),
    monai.transforms.GridPatchd(
        keys=["input", "target"],
        patch_size=(96, 96, 96), 
        pad_mode='reflect')
    ])

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
    