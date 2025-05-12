import torch

from src.multimodal import Multimodal
from src.strongly_convex.diagonal_quadratic import DiagonalQuadratic

class SumOfDiagonalGaussian(Multimodal):
    def __init__(self, diagonals, offsets, weights, inverse=False) -> None:
        """
        
        :param diagonals: m x d
        :param offsets: m x d or [None, ..., None]
        :param weights: m
        """
        super().__init__([DiagonalQuadratic(diagonals[i], offset=offsets[i], inverse=inverse) for i in range(len(diagonals))], weights / torch.sqrt(torch.prod(diagonals,1)))