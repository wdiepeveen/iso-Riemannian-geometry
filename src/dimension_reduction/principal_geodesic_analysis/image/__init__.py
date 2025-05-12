import torch
from src.dimension_reduction.principal_geodesic_analysis import l2PGASolver

class l2PGAImageSolver(l2PGASolver):
    def __init__(self, data, image_euclidean, base_point):
        super().__init__(data.shape[0], torch.prod(torch.tensor(data.shape[1:])).item(), data, image_euclidean, base_point)

        _, in_channels, height, width = data.shape
        self.C = in_channels
        self.H = height
        self.W = width