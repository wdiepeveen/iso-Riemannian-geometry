from src.manifolds.euclidean.sequence import SequenceEuclidean
from src.manifolds.euclidean.sequence_from_vector.standard import StandardSequenceFromVectorEuclidean

class StandardSequenceEuclidean(SequenceEuclidean):
    """ Base class describing Euclidean space of dimension d """

    def __init__(self, in_channels, length):
        super().__init__(in_channels, length)

        self.sequence_from_vector_euclidean = StandardSequenceFromVectorEuclidean(in_channels, length)

    def inner(self, x, X, Y):
        """

        :param x: N x (C x L) 
        :param X: N x M x (C x L)
        :param Y: N x L x (C x L)
        :return: N x M x L
        """
        return self.sequence_from_vector_euclidean.inner(x, X, Y)
    
    def norm(self, x, X):
        """

        :param x: N x (C x L) 
        :param X: N x M x (C x L) 
        :return: N x M
        """
        return self.sequence_from_vector_euclidean.norm(x, X)

    def barycentre(self, x, tol=None, max_iter=None, step_size=None, red_coef=None):
        """

        :param x: N x (C x L)
        :return: (C x L)
        """
        return self.sequence_from_vector_euclidean.barycentre(x, tol=tol, max_iter=max_iter, step_size=step_size, red_coef=red_coef)
    
    def geodesic(self, x, y, t):
        """

        :param x: N x M x (C x L) 
        :param y: N x L x (C x L) 
        :param t: K or N x M x L x K
        :return: N x M x L x K x (C x L)
        """
        return self.sequence_from_vector_euclidean.geodesic(x, y, t)
            

    def log(self, x, y):
        """

        :param x: N x M x (C x L)
        :param y: N x L x (C x L)
        :return: N x M x L x (C x L)
        """
        return self.sequence_from_vector_euclidean.log(x, y)

    def exp(self, x, X):
        """

        :param x: N x (C x L)
        :param X: N x M x (C x L)
        :return: N x M x (C x L)
        """
        return self.sequence_from_vector_euclidean.exp(x, X)
    
    def distance(self, x, y):
        """

        :param x: N x M x (C x L)
        :param y: N x L x (C x L)
        :return: N x M x L
        """
        return self.sequence_from_vector_euclidean.distance(x, y)

    def parallel_transport(self, x, X, y):
        """

        :param x: N x M x (C x L)
        :param X: N x M x L x K x (C x L)
        :param y: N x L x (C x L)
        :return: N x M x L x K x (C x L)
        """
        return self.sequence_from_vector_euclidean.parallel_transport(x, X, y)
    
