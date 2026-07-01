from src.manifolds.euclidean.image import ImageEuclidean
from src.manifolds.euclidean.image_from_vector.standard import StandardImageFromVectorEuclidean

class StandardImageEuclidean(ImageEuclidean):
    """ Base class describing Euclidean space of dimension d """

    def __init__(self, in_channels, height, width):
        super().__init__(in_channels, height, width)

        self.image_from_vector_euclidean = StandardImageFromVectorEuclidean(in_channels, height, width)

    def inner(self, x, X, Y):
        """

        :param x: N x (C x H x W) 
        :param X: N x M x (C x H x W)
        :param Y: N x L x (C x H x W)
        :return: N x M x L
        """
        return self.image_from_vector_euclidean.inner(x, X, Y)
    
    def norm(self, x, X):
        """

        :param x: N x (C x H x W) 
        :param X: N x M x (C x H x W) 
        :return: N x M
        """
        return self.image_from_vector_euclidean.norm(x, X)

    def barycentre(self, x, tol=None, max_iter=None, step_size=None, red_coef=None):
        """

        :param x: N x (C x H x W)
        :return: (C x H x W)
        """
        return self.image_from_vector_euclidean.barycentre(x, tol=tol, max_iter=max_iter, step_size=step_size, red_coef=red_coef)
    
    def geodesic(self, x, y, t):
        """

        :param x: N x M x (C x H x W) 
        :param y: N x L x (C x H x W) 
        :param t: K or N x M x L x K
        :return: N x M x L x K x (C x H x W)
        """
        return self.image_from_vector_euclidean.geodesic(x, y, t)
            

    def log(self, x, y):
        """

        :param x: N x M x (C x H x W)
        :param y: N x L x (C x H x W)
        :return: N x M x L x (C x H x W)
        """
        return self.image_from_vector_euclidean.log(x, y)

    def exp(self, x, X):
        """

        :param x: N x (C x H x W)
        :param X: N x M x (C x H x W)
        :return: N x M x (C x H x W)
        """
        return self.image_from_vector_euclidean.exp(x, X)
    
    def distance(self, x, y):
        """

        :param x: N x M x (C x H x W)
        :param y: N x L x (C x H x W)
        :return: N x M x L
        """
        return self.image_from_vector_euclidean.distance(x, y)

    def parallel_transport(self, x, X, y):
        """

        :param x: N x M x (C x H x W)
        :param X: N x M x K x (C x H x W)
        :param y: N x L x (C x H x W)
        :return: N x M x L x K x (C x H x W)
        """
        return self.image_from_vector_euclidean.parallel_transport(x, X, y)
    
