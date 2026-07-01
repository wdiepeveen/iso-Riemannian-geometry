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

    def barycentre(self, x, tol=None, max_iter=None, step_size=None, red_coef=None):
        """

        :param x: N x d
        :return: d
        """
        return torch.mean(x, 0)
    
    def geodesic(self, x, y, t):
        """

        :param x: N x M x d
        :param y: N x L x d
        :param t: K or N x M x L x K
        :return: N x M x L x K x d
        """
        if t.ndim == 1:
            return (1 - t[None,None,None,:,None]) * x[:,:,None,None] + t[None,None,None,:,None] * y[:,None,:,None]  
        else:
            return (1 - t[:, :, :, :, None]) * x[:, :, None, None] + t[:, :, :, :, None] * y[:, None, :, None]

    def log(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L x d
        """
        return y[:,None,:] - x[:,:,None]

    def exp(self, x, X):
        """

        :param x: N x d
        :param X: N x M x d
        :return: N x M x d
        """
        return x[:,None] + X
    
    def distance(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        return torch.sqrt(torch.sum((x[:,:,None] - y[:,None,:]) ** 2, -1))

    def parallel_transport(self, x, X, y):
        """

        :param x: N x M x d
        :param X: N x M x K x d
        :param y: N x L x d
        :return: N x M x L x K x d
        """
        return X[:,:,None,:].repeat(1, 1, y.shape[1], 1, 1)
    