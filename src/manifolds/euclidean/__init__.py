from src.manifolds import Manifold

class Euclidean(Manifold):
    def __init__(self, d):
        super().__init__(d)

    def metric_tensor(self, x):
        """
        :param x: N x [Epoint]
        :return: N x [Evector] x [Evector]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def inverse_metric_tensor(self, x):
        """
        :param x: N x [Epoint]
        :return: N x [Evector] x [Evector]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )