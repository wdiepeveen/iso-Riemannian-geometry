from src.manifolds.euclidean.sequence_from_vector import SequenceFromVectorEuclidean
from src.manifolds.euclidean.vector.standard import StandardVectorEuclidean

class StandardSequenceFromVectorEuclidean(SequenceFromVectorEuclidean):
    """ Base class describing Euclidean space of dimension d """

    def __init__(self, in_channels, length):
        super().__init__(in_channels, length)

        self.vector_euclidean = StandardVectorEuclidean(in_channels * length)

    