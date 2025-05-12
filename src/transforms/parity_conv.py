import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.transforms import Transform

class ParityConv2DTransform(Transform):
    def __init__(self, in_channels, height, width, kernel_size, bias=True, parity_equivariance=True, parity=0, unit_det=True):
        assert (kernel_size - 1) % 2 == 0
        super().__init__()
        self.C = in_channels
        self.H = height
        self.W = width
        self.K = kernel_size
        self.P = self.K // 2

        self.parity_equivariance = parity_equivariance # If True, the network is equivariant for shifts that are multiples 2 (so not equivariant for all shits)
        
        self.diagonal = nn.Parameter(torch.ones(self.C)[:,None,None])
        self.conv2d = nn.Conv2d(self.C, self.C, self.K, padding=self.P, bias=False)
        if bias:
            if self.parity_equivariance:
                self.bias = nn.Parameter(torch.randn(self.C)[:,None,None])
            else:
                self.bias = nn.Parameter(torch.randn(self.C, self.H, self.W))
        else:
            self.bias = torch.zeros(self.C, self.H, self.W)
        
        self.parity = parity % 2
        self.weight_mask = self.generate_weight_mask()
        self.image_mask = self.generate_image_mask()
        
        self.unit_det = unit_det

    def forward(self, x, context=None):
        z_1 = x.clone()
        # Multiply by diagonal
        if self.unit_det:
            z_2 = z_1
            log_abs_det = torch.zeros(1, device=x.device)
        else:
            diagonal = F.softplus(self.diagonal)
            z_2 = z_1 * diagonal
            log_abs_det = torch.log(diagonal.repeat(1,self.H,self.W)[self.image_mask.bool()]).sum().to(x.device) 

        # Convolve
        weight = self.conv2d.weight * self.weight_mask.to(x.device)
        z_3 = nn.functional.conv2d(z_2, weight, self.conv2d.bias, self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups)
        z_4 = torch.zeros_like(z_1)
        z_4[:,~self.image_mask.bool()] = z_2[:,~self.image_mask.bool()]
        z_4[:,self.image_mask.bool()] = z_2[:,self.image_mask.bool()] + z_3[:,self.image_mask.bool()]

        # Add bias
        z_5 = z_4 + self.bias.to(x.device)
        return z_5, log_abs_det.expand(x.shape[0])
    
    def inverse(self, z, context=None):
        x_1 = z.clone()
        # Subtract bias
        x_2 = x_1 - self.bias.to(z.device)

        # Convolve
        weight = self.conv2d.weight * self.weight_mask.to(z.device)
        x_3 = nn.functional.conv2d(x_2, weight, self.conv2d.bias, self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups)
        x_4 = torch.zeros_like(x_1)
        x_4[:,~self.image_mask.bool()] = x_2[:,~self.image_mask.bool()]
        x_4[:,self.image_mask.bool()] = x_2[:,self.image_mask.bool()] - x_3[:,self.image_mask.bool()]

        # Divide by diagonal
        if self.unit_det:
            x_5 = x_4
            log_abs_det = torch.zeros(1, device=z.device)
        else:
            diagonal = F.softplus(self.diagonal)
            x_5 = x_4 / diagonal
            log_abs_det = -torch.log(diagonal.repeat(1,self.H,self.W)[self.image_mask.bool()]).sum().to(z.device) 

        return x_5, log_abs_det.expand(z.shape[0])
    
    def generate_weight_mask(self):
        mask = torch.zeros(self.C, self.C, self.K, self.K)
        for i in range(self.K):
            for j in range(self.K):
                mask[:, :, i, j] = 0 if (i + j) % 2 == 0 else 1
        return mask
    
    def generate_image_mask(self):
        mask = torch.zeros(self.C, self.H, self.W)
        for i in range(self.H):
            for j in range(self.W):
                mask[:, i, j] = 1 if (i + j) % 2 == self.parity else 0
        return mask
        