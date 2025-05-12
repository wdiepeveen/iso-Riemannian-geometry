import torch
import torch.nn as nn

from nflows.transforms import Transform

from src.nn.module.conv2d.masked_conv import MaskedConv2d

class MultiLayerParityConv2DTransform(Transform):
    def __init__(self, in_channels, height, width, kernel_size, latent_channels, activation_class, activation_args=None, parity=0):
        assert (kernel_size - 1) % 2 == 0
        super().__init__()
        self.C = in_channels
        self.H = height
        self.W = width
        self.K = kernel_size
        self.P = self.K // 2
        self.L = latent_channels

        self.parity = parity % 2
        self.image_mask = self.generate_image_mask()

        self.cnn = nn.Sequential(
            *[
                MaskedConv2d(self.C, self.L, self.K, padding=self.P, bias=True), # False
                activation_class(self.L, **activation_args),
                MaskedConv2d(self.L, self.L, self.K, padding=self.P, bias=True),
                activation_class(self.L, **activation_args),
                MaskedConv2d(self.L, self.C, self.K, padding=self.P, bias=True)
            ]
        )

    def forward(self, x, context=None):
        x_ = x.clone()
        log_abs_det = torch.zeros(1, device=x.device)
        
        # Convolve
        c = self.cnn(x_)
        
        # Apply non-linearity
        z = torch.zeros_like(x_)
        z[:,~self.image_mask.bool()] = x_[:,~self.image_mask.bool()]
        z[:,self.image_mask.bool()] = x_[:,self.image_mask.bool()] + c[:,self.image_mask.bool()]

        return z, log_abs_det.expand(x.shape[0])
    
    def inverse(self, z, context=None):
        z_ = z.clone()
        log_abs_det = torch.zeros(1, device=z.device)
        
        # Convolve
        c = self.cnn(z_)
        
        # Apply non-linearity
        x = torch.zeros_like(z_)
        x[:,~self.image_mask.bool()] = z_[:,~self.image_mask.bool()]
        x[:,self.image_mask.bool()] = z_[:,self.image_mask.bool()] - c[:,self.image_mask.bool()]

        return x, log_abs_det.expand(z.shape[0])
    
    def generate_image_mask(self):
        mask = torch.zeros(self.C, self.H, self.W)
        for i in range(self.H):
            for j in range(self.W):
                mask[:, i, j] = 1 if (i + j) % 2 == self.parity else 0
        return mask
        