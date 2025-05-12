import torch

from src.strongly_convex import StronglyConvex

class DiagonalQuadratic(StronglyConvex): 
    """ Class that implements the strongly convex function x \mapsto 1/2 x^\top A^{-1} x, where A is diagonal with positive entries """
    def __init__(self, diagonal, offset=None, inverse=False) -> None:
        super().__init__(len(diagonal))

        if inverse == False:
            self.inverse_diagonal = 1 / diagonal # torch tensor of size d
        else:
            self.inverse_diagonal = diagonal
        if offset is None:
            self.offset = torch.zeros_like(diagonal)
        else:
            self.offset = offset

    def forward(self, x):
        """
        :param x: N x d
        :return: N
        """
        return 0.5 * torch.sum(self.inverse_diagonal[None] * (x - self.offset[None]) **2, 1)
    
    def grad_forward(self, x):
        """
        :param x: N x d
        :return: N x d
        """
        return self.inverse_diagonal[None] * (x - self.offset[None])
    
    def differential_grad_forward(self, x, X):
        """
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        return self.inverse_diagonal[None] * X