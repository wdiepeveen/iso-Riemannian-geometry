import torch

from src.manifolds.isometrized_euclidean import l2IsometrizedEuclidean

class l2IsometrizedSequenceEuclidean(l2IsometrizedEuclidean):
    def __init__(self, sequence_euclidean, num_intervals=10):
        super().__init__(sequence_euclidean, num_intervals=num_intervals)

    def l2_inner(self, X, Y):
        """

        :param X: N x M x (C x L) 
        :param Y: N x L x (C x L)
        :return: N x M x L
        """
        return torch.einsum("NMcl,NLcl->NML", X, Y)
    
    def l2_norm(self, X):
        """

        :param X: N x M x (C x L)
        :return: N x M
        """
        return torch.einsum("NMcl,NMcl->NM", X, X).sqrt()

    