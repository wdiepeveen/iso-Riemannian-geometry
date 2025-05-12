import torch

from src.dimension_reduction.principal_geodesic_analysis.vector import l2PGAVectorSolver

class l2TangentSpacePCAVectorSolver(l2PGAVectorSolver):
    def __init__(self, data, vector_euclidean, base_point) -> None:
        super().__init__(data, vector_euclidean, base_point)

        # compute svd
        self.U, self.Sigma, self.V = torch.svd(self.log_x_data)

    def get_Xi(self, rank):
        assert rank > 0 and rank <= min(self.d, self.N)
        return torch.einsum("Nk,k,dk->Nd", self.U[:,0:rank], self.Sigma[0:rank], self.V[:,0:rank])
