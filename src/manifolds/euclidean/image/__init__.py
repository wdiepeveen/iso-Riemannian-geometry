from src.manifolds.euclidean import Euclidean

class ImageEuclidean(Euclidean):
    def __init__(self, in_channels, height, width):
        super().__init__(in_channels * height * width)
        self.C = in_channels
        self.H = height
        self.W = width