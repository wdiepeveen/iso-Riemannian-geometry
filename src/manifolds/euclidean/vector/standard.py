import torch

from src.manifolds.euclidean.vector import VectorEuclidean

class StandardVectorEuclidean(VectorEuclidean):
    """ Base class describing Euclidean space of dimension d """

    def __init__(self, d):
        super().__init__(d)

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

    def barycentre(self, x):
        """

        :param x: N x d
        :return: d
        """
        return torch.mean(x, 0)
    
    def geodesic(self, x, y, t):
        """

        :param x: d or N x d
        :param y: d or N x d
        :param t: N or 1
        :return: N x d
        """
        if len(t) == 1:
            return (1 - t) * x + t * y
        else:
            return (1 - t[:,None]) * x[None] + t[:,None] * y[None]
            

    def log(self, x, y):
        """

        :param x: d
        :param y: N x d
        :return: N x d
        """
        return y - x

    def exp(self, x, X):
        """

        :param x: d
        :param X: N x d
        :return: N x d
        """
        return x + X
    
    def distance(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        return torch.sqrt(torch.sum((x[:,:,None] - y[:,None,:]) ** 2, -1) + 1e-8)

    def parallel_transport(self, x, X, y):
        """

        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        return X
    
    def metric_tensor(self, x):
        """
        :return: N x d x d
        """
        N, _ = x.shape
        return torch.diag_embed(torch.ones(self.d))[None] * torch.ones(N)[:,None,None]
    
    def inverse_metric_tensor(self, x):
        """
        :return: N x d x d
        """
        N, _ = x.shape
        return torch.diag_embed(torch.ones(self.d))[None] * torch.ones(N)[:,None,None]
