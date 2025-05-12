import torch

from src.manifolds.isometrized_euclidean import l2IsometrizedEuclidean

class l2IsometrizedVectorEuclidean(l2IsometrizedEuclidean):
    def __init__(self, vector_euclidean, num_intervals=100):
        super().__init__(vector_euclidean, num_intervals=num_intervals)

    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        return torch.einsum("NMi,NLi->NML", X, Y)
    
    def norm(self, x, X):
        """

        :param x: N x d
        :param X: N x M x d
        :return: N x M
        """
        return torch.einsum("NMi,NMi->NM", X, X).sqrt()