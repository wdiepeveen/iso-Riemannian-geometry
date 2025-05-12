import torch

from src.time_flows import TimeFlow

class PiecewiseLinearTimeFlow(TimeFlow):

    def __init__(self, z) -> None:
        super().__init__()

        self.num_bins = int(z.shape[0])-1
        interval_lengths = ((z[1:] - z[:-1])**2).sum([i for i in range(1,len(z.shape))]).sqrt()

        self.tau = torch.cat([torch.tensor([0.], device=z.device), torch.cumsum(interval_lengths,0) / torch.sum(interval_lengths)])

    def forward(self, t):
        return torch.clamp(torch.sum(1 / self.num_bins * torch.clamp((t[:,None] - self.tau[None,:-1]) / (self.tau[None,1:] - self.tau[None,:-1]), 0., 1.),1),0.,1.)
    
    def differential_forward(self, t):
        def forwards(t):
            return torch.sum(self.forward(t))
        output = torch.autograd.functional.jacobian(forwards, t)
        return output