import torch
import torch.nn as nn

from nflows.transforms import Transform

class HouseholderLinearTransform(Transform):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.v = nn.Parameter(torch.randn(self.features) / self.features**(1/2))

    def forward(self, x, context=None):
        v = self.v / (self.v.norm() + 1e-8)
        log_abs_det = torch.zeros(1, device=x.device)
        out = x - 2 * v[None] * (x * v[None]).sum(1)[:,None]
        return out, log_abs_det.expand(x.shape[0])
        
    def inverse(self, z, context=None):
        return self.forward(z)
