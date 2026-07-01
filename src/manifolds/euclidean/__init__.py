from src.manifolds import Manifold

class Euclidean(Manifold):
    def __init__(self, d):
        super().__init__(d)