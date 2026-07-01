import torch

from src.diffeomorphisms.vector.affine_unbend import AffineUnbendVectorDiffeomorphism
from src.manifolds.euclidean.vector.pullback.standard import StandardPullbackVectorEuclidean

class AffineUnbendStandardPullbackVectorEuclidean(StandardPullbackVectorEuclidean):

    def __init__(self, angle=torch.pi/4, eta=4.):
        super().__init__(AffineUnbendVectorDiffeomorphism(angle, eta))