import torch
import torch.nn as nn

from src.nn.module.activation import Activation

class PolynomialActivation(Activation):
    def __init__(self, num_features, order=2):
        super().__init__(num_features)
        self.order = order
        self.coefficients = nn.Parameter(torch.zeros(self.d, self.order+1))

    def forward(self, x):
        powers = torch.arange(self.order+1, device=x.device)
        if x.dim() == 2:  # Vector data (batch, features)
            x_powers = x.unsqueeze(-1) ** powers
            return torch.sum(self.coefficients * x_powers, dim=-1)
        elif x.dim() == 4:  # Image data (batch, channels, height, width)
            assert x.shape[1] == self.d
            x_powers = x.unsqueeze(-1) ** powers  # (batch, channels, height, width, order)
            coefficients = self.coefficients.view(self.d, 1, 1, self.order+1)
            return torch.sum(coefficients * x_powers, dim=-1)
        else:
            raise ValueError("Input must be 2D (vector) or 4D (image)")
