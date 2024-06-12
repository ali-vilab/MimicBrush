from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin
import torch


class Conv2d(nn.Conv2d):
    def forward(self, x):
        x = super().forward(x)
        return x


class DepthGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int=4,
        conditioning_channels: int = 1,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )
        self.conv_out = Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
        )

    def forward(self, conditioning):
        conditioning = F.interpolate(conditioning, size=(512,512), mode = 'bilinear', align_corners=True)
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding