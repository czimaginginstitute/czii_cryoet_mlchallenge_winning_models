import torch
from torch import nn
from monai.networks.nets.flexible_unet import SegmentationHead, UNetDecoder, FLEXUNET_BACKBONE


class PatchedUNetDecoder(UNetDecoder):
    
    """UNet decoder to output results of each feature"""
    
    def forward(
            self, features: list[torch.Tensor], 
            skip_connect: int = 4
        ):
        skips = features[:-1][::-1] # skip the last channel, [E3, E2, E1, E0]
        features = features[1:][::-1] # skip the first channel, [E4, E3, E2, E1]

        out = []
        x = features[0]
        out += [x]
        for i, block in enumerate(self.blocks):
            if i < skip_connect: # residual for the top 4 layers
                skip = skips[i]
            else:
                skip = None
            x = block(x, skip)
            out += [x]
        return out

class FlexibleUNet(nn.Module):
    """
    A flexible implementation of UNet-like encoder-decoder architecture. 
    
    (Adjusted to support PatchDecoder and multi segmentation heads)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str,
        pretrained: bool = False,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        spatial_dims: int = 2,
        norm: str | tuple = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: str | tuple = ("relu", {"inplace": True}),
        dropout: float | tuple = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        pre_conv: str = "default",
        interp_mode: str = "nearest",
        is_pad: bool = True,
    ) -> None:
        """
        A flexible implement of UNet, in which the backbone/encoder can be replaced with
        any efficient or residual network. Currently the input must have a 2 or 3 spatial dimension
        and the spatial size of each dimension must be a multiple of 32 if is_pad parameter
        is False.
        Please notice each output of backbone must be 2x downsample in spatial dimension
        of last output. For example, if given a 512x256 2D image and a backbone with 4 outputs.
        Spatial size of each encoder output should be 256x128, 128x64, 64x32 and 32x16.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels. out_channels = nclasses + 1. background is added as one of the classes. 
            backbone: name of backbones to initialize, only support efficientnet and resnet right now,
                can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2, resnet10, ..., resnet200].
            pretrained: whether to initialize pretrained weights. ImageNet weights are available for efficient networks
                if spatial_dims=2 and batch norm is used. MedicalNet weights are available for residual networks
                if spatial_dims=3 and in_channels=1. Default to False.
            decoder_channels: number of output channels for all feature maps in decoder.
                `len(decoder_channels)` should equal to `len(encoder_channels) - 1`,default
                to (256, 128, 64, 32, 16).
            spatial_dims: number of spatial dimensions, default to 2.
            norm: normalization type and arguments, default to ("batch", {"eps": 1e-3,
                "momentum": 0.1}).
            act: activation type and arguments, default to ("relu", {"inplace": True}).
            dropout: dropout ratio, default to 0.0.
            decoder_bias: whether to have a bias term in decoder's convolution blocks.
            upsample: upsampling mode, available options are``"deconv"``, ``"pixelshuffle"``,
                ``"nontrainable"``.
            pre_conv:a conv block applied before upsampling. Only used in the "nontrainable" or
                "pixelshuffle" mode, default to `default`.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            is_pad: whether to pad upsampling features to fit features from encoder. Default to True.
                If this parameter is set to "True", the spatial dim of network input can be arbitrary
                size, which is not supported by TensorRT. Otherwise, it must be a multiple of 32.
        """
        super().__init__()

        if backbone not in FLEXUNET_BACKBONE.register_dict:
            raise ValueError(
                f"invalid model_name {backbone} found, must be one of {FLEXUNET_BACKBONE.register_dict.keys()}."
            )

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")

        encoder = FLEXUNET_BACKBONE.register_dict[backbone]
        self.backbone = backbone
        self.spatial_dims = spatial_dims
        encoder_parameters = encoder["parameter"]
        if not (
            ("spatial_dims" in encoder_parameters)
            and ("in_channels" in encoder_parameters)
            and ("pretrained" in encoder_parameters)
        ):
            raise ValueError("The backbone init method must have spatial_dims, in_channels and pretrained parameters.")
        encoder_feature_num = encoder["feature_number"]
        if encoder_feature_num > 5:
            raise ValueError("Flexible unet can only accept no more than 5 encoder feature maps.")

        decoder_channels = decoder_channels[:encoder_feature_num]
        self.skip_connect = encoder_feature_num - 1
        encoder_parameters.update({"spatial_dims": spatial_dims, "in_channels": in_channels, "pretrained": pretrained})
        encoder_channels = tuple([in_channels] + list(encoder["feature_channel"]))
        encoder_type = encoder["type"]
        self.encoder = encoder_type(**encoder_parameters)
        
        self.decoder = PatchedUNetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=pre_conv,
            align_corners=None,
            is_pad=is_pad,
        )
        self.segmentation_heads = nn.ModuleList([SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channel,
            out_channels=out_channels + 1,
            kernel_size=3,
            act=None,
        ) for decoder_channel in decoder_channels[:-1]])

    def forward(self, inputs: torch.Tensor):

        x = inputs
        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out, self.skip_connect)[1:-1]  # skip the first and the last feature 
        x_seg = [self.segmentation_heads[i](decoder_out[i]) for i in range(len(decoder_out))]

        return x_seg
