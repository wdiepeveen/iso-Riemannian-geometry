import torch

from src.dimension_reduction.principal_geodesic_analysis.image import l2PGAImageSolver

class l2TangentSpacePCAImageSolver(l2PGAImageSolver):
    def __init__(self, data, image_euclidean, base_point) -> None:
        super().__init__(data, image_euclidean, base_point)

        # compute svd
        self.U, self.Sigma, self.V = torch.svd(self.log_x_data.reshape(-1, self.d))

    def get_Xi(self, rank):
        assert rank > 0 and rank <= min(self.d, self.N)
        return torch.einsum("Nk,k,dk->Nd", self.U[:,0:rank], self.Sigma[0:rank], self.V[:,0:rank]).reshape(-1, self.C, self.H, self.W)
