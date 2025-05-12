import torch

from src.manifolds.isometrized_euclidean import l2IsometrizedEuclidean

class l2IsometrizedImageEuclidean(l2IsometrizedEuclidean):
    def __init__(self, image_euclidean, num_intervals=100):
        super().__init__(image_euclidean, num_intervals=num_intervals)

    def inner(self, x, X, Y):
        """

        :param x: N x (C x H x W) 
        :param X: N x M x (C x H x W) 
        :param Y: N x L x (C x H x W) 
        :return: N x M x L
        """
        return torch.einsum("NMchw,NLchw->NML", X, Y)
    
    def norm(self, x, X):
        """

        :param x: N x (C x H x W) 
        :param X: N x M x (C x H x W) 
        :return: N x M
        """
        return torch.einsum("NMchw,NMchw->NM", X, X).sqrt()

    