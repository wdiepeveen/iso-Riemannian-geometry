import torch

from src.manifolds.euclidean import Euclidean

class VectorEuclidean(Euclidean):
    def __init__(self, d):
        super().__init__(d)