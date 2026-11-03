import torch.nn as nn

from src.transforms.image.parity_conv import ParityConvImageTransform

class SimpleTanh2DParityConvImageTransform(ParityConvImageTransform):
    def __init__(self, in_channels, height, width, hidden_channels, kernel_size, parity=0):
        assert (kernel_size - 1) % 2 == 0
        conv_network = nn.Sequential(
            *[
                nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.Tanh(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.Tanh(),
                nn.Conv2d(hidden_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)
            ]
        )
        super().__init__(in_channels, height, width, conv_network, parity)