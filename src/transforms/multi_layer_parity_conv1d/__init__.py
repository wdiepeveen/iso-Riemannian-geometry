import torch
import torch.nn as nn

from nflows.transforms import Transform

from src.nn.module.conv1d.masked_conv import MaskedConv1d

class MultiLayerParityConv1DTransform(Transform):
    def __init__(self, in_channels, length, kernel_size, latent_channels, activation_class, activation_args=None, parity=0):
        assert (kernel_size - 1) % 2 == 0
        super().__init__()
        self.C = in_channels
        self.L = length
        self.K = kernel_size
        self.M = latent_channels

        self.parity = parity % 2
        self.mask = self.generate_sequence_mask()

        self.conv = nn.Sequential(
            *[
                nn.Conv1d(self.C, self.M, self.K, padding=self.K//2, bias=True), # False
                activation_class(self.M, **activation_args),
                nn.Conv1d(self.M, self.M, self.K, padding=self.K//2, bias=True),
                activation_class(self.M, **activation_args),
                nn.Conv1d(self.M, self.C, self.K, padding=self.K//2, bias=True)
            ]
        )

    def forward(self, x, context=None):
        log_abs_det = torch.zeros(1, device=x.device)
        
        # Apply non-linearity
        z = torch.zeros_like(x)
        z[:,self.mask] = x[:,self.mask]
        z[:,~self.mask] = x[:,~self.mask] + self.conv(self.mask[None].to(x.device) * x)[:,~self.mask]
        return z, log_abs_det.expand(x.shape[0])
    
    def inverse(self, z, context=None):
        log_abs_det = torch.zeros(1, device=z.device)
    
        # Apply non-linearity
        x = torch.zeros_like(z)
        x[:,self.mask] = z[:,self.mask]
        x[:,~self.mask] = z[:,~self.mask] - self.conv(self.mask[None].to(z.device) * z)[:,~self.mask]
        return x, log_abs_det.expand(z.shape[0])
    
    def generate_sequence_mask(self):
        mask = torch.zeros(self.C, self.L, dtype=torch.bool)
        for i in range(self.L):
            mask[:, i] = 1 if i % 2 == self.parity else 0
        return mask
        