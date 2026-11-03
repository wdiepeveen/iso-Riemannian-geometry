import torch

from src.transforms.image import ImageTransform

class ParityConvImageTransform(ImageTransform):
    def __init__(self, in_channels, height, width, conv_network, parity):
        super().__init__(in_channels, height, width)
        self.conv = conv_network
        self.parity = parity % 2

        self.mask = self.generate_image_mask()

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
    
    def generate_image_mask(self):
        mask = torch.zeros(self.C, self.H, self.W, dtype=torch.bool)
        for i in range(self.H):
            for j in range(self.W):
                mask[:, i, j] = 1 if (i + j) % 2 == self.parity else 0
        return mask