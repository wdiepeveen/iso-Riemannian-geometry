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

    def barycentre(self, x):
        """

        :param x: N x (C x H x W)
        :return: (C x H x W)
        """
        return self.image_from_vector_euclidean.barycentre(x)
    
    def geodesic(self, x, y, t):
        """

        :param x: (C x H x W) or N x (C x H x W)
        :param y: (C x H x W) or N x (C x H x W)
        :param t: N or 1
        :return: N x (C x H x W)
        """
        return self.image_from_vector_euclidean.geodesic(x, y, t)
            

    def log(self, x, y):
        """

        :param x: (C x H x W)
        :param y: N x (C x H x W)
        :return: N x (C x H x W)
        """
        return self.image_from_vector_euclidean.log(x, y)

    def exp(self, x, X):
        """

        :param x: (C x H x W)
        :param X: N x (C x H x W)
        :return: N x (C x H x W)
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

        :param x: (C x H x W)
        :param X: N x (C x H x W)
        :param y: (C x H x W)
        :return: N x (C x H x W)
        """
        return self.image_from_vector_euclidean.parallel_transport(x, X, y)
    
    def metric_tensor(self, x):
        """
        :return: N x (C x H x W) x (C x H x W)
        """
        return self.image_from_vector_euclidean.metric_tensor(x)
    
    def inverse_metric_tensor(self, x):
        """
        :return: N x (C x H x W) x (C x H x W)
        """
        N = x.shape[0]
        return self.image_from_vector_euclidean.inverse_metric_tensor(x)