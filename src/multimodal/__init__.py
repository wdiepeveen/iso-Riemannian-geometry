import torch

class Multimodal:
    def __init__(self, strongly_convexs, weights) -> None:
        self.d = strongly_convexs[0].d
        self.psi = strongly_convexs
        self.weights = weights
        self.m = len(strongly_convexs)
    

    def log_density(self, x):
        N = x.shape[0]
        psi_phi_x = torch.zeros((N,self.m))
        for i in range(self.m):
            psi_phi_x[:,i] = self.psi[i].forward(x)
        result = torch.log(torch.sum(self.weights[None] * torch.exp(-psi_phi_x),-1))
        return result