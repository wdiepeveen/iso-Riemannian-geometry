from src.manifolds.euclidean.image_from_vector import ImageFromVectorEuclidean
from src.manifolds.euclidean.vector.standard import StandardVectorEuclidean

class StandardImageFromVectorEuclidean(ImageFromVectorEuclidean):
    """ Base class describing Euclidean space of dimension d """

    def __init__(self, in_channels, height, width):
        super().__init__(in_channels, height, width)

        self.vector_euclidean = StandardVectorEuclidean(in_channels * height * width)

    