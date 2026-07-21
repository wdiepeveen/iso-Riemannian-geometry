from src.manifolds.euclidean import Euclidean

class SequenceEuclidean(Euclidean):
    def __init__(self, in_channels, length):
        super().__init__(in_channels * length)
        self.C = in_channels
        self.L = length