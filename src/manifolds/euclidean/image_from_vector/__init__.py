from src.manifolds.euclidean import Euclidean

class ImageFromVectorEuclidean(Euclidean):
    def __init__(self, in_channels, height, width):
        super().__init__(in_channels * height * width)
        self.C = in_channels
        self.H = height
        self.W = width
        self.vector_euclidean = None

    def inner(self, x, X, Y):
        """

        :param x: N x (C x H x W) 
        :param X: N x M x (C x H x W)
        :param Y: N x L x (C x H x W)
        :return: N x M x L
        """
        N = x.shape[0]
        M = X.shape[1]
        L = Y.shape[1]
        return self.vector_euclidean.inner(
            x.reshape(N, self.C * self.H * self.W), 
            X.reshape(N, M, self.C * self.H * self.W),
            Y.reshape(N, L, self.C * self.H * self.W)
            )
    
    def norm(self, x, X):
        """

        :param x: N x (C x H x W) 
        :param X: N x M x (C x H x W) 
        :return: N x M
        """
        N = x.shape[0]
        M = X.shape[1]
        return self.vector_euclidean.norm(
            x.reshape(N, self.C * self.H * self.W), 
            X.reshape(N, M, self.C * self.H * self.W)
            )

    def barycentre(self, x):
        """

        :param x: N x (C x H x W)
        :return: (C x H x W)
        """
        N = x.shape[0]
        return self.vector_euclidean.barycentre(x.reshape(N, self.C * self.H * self.W)).reshape(self.C, self.H, self.W)
    
    def geodesic(self, x, y, t):
        """

        :param x: (C x H x W) or N x (C x H x W)
        :param y: (C x H x W) or N x (C x H x W)
        :param t: N or 1
        :return: N x (C x H x W)
        """
        if len(t) == 1:
            N = x.shape[0]
            return self.vector_euclidean.geodesic(
                x.reshape(N, self.C * self.H * self.W),
                y.reshape(N, self.C * self.H * self.W),
                t
            ).reshape(N, self.C, self.H, self.W)
        else:
            N = len(t)
            return self.vector_euclidean.geodesic(
                x.reshape(self.C * self.H * self.W),
                y.reshape(self.C * self.H * self.W),
                t
            ).reshape(N, self.C, self.H, self.W)
            

    def log(self, x, y):
        """

        :param x: (C x H x W)
        :param y: N x (C x H x W)
        :return: N x (C x H x W)
        """
        N = y.shape[0]
        return self.vector_euclidean.log(
            x.reshape(self.C * self.H * self.W),
            y.reshape(N, self.C * self.H * self.W)
            ).reshape(N, self.C, self.H, self.W)

    def exp(self, x, X):
        """

        :param x: (C x H x W)
        :param X: N x (C x H x W)
        :return: N x (C x H x W)
        """
        N = X.shape[0]
        return self.vector_euclidean.exp(
            x.reshape(self.C * self.H * self.W),
            X.reshape(N, self.C * self.H * self.W)
            ).reshape(N, self.C, self.H, self.W)
    
    def distance(self, x, y):
        """

        :param x: N x M x (C x H x W)
        :param y: N x L x (C x H x W)
        :return: N x M x L
        """
        N, M = x.shape[0:2]
        L = y.shape[1]
        return self.vector_euclidean.distance(
            x.reshape(N, M, self.C * self.H * self.W),
            y.reshape(N, L, self.C * self.H * self.W)
        )

    def parallel_transport(self, x, X, y):
        """

        :param x: (C x H x W)
        :param X: N x (C x H x W)
        :param y: (C x H x W)
        :return: N x (C x H x W)
        """
        N = X.shape[0]
        return self.vector_euclidean.parallel_transport(
            x.reshape(self.C * self.H * self.W),
            X.reshape(N, self.C * self.H * self.W),
            y.reshape(self.C * self.H * self.W)
        ).reshape(N, self.C, self.H, self.W)
    
    def metric_tensor(self, x):
        """
        :return: N x (C x H x W) x (C x H x W)
        """
        N = x.shape[0]
        return self.vector_euclidean.metric_tensor(x.reshape(N, self.C * self.H * self.W)).reshape(N, self.C, self.H, self.W, self.C, self.H, self.W)
    
    def inverse_metric_tensor(self, x):
        """
        :return: N x (C x H x W) x (C x H x W)
        """
        N = x.shape[0]
        return self.vector_euclidean.inverse_metric_tensor(x.reshape(N, self.C * self.H * self.W)).reshape(N, self.C, self.H, self.W, self.C, self.H, self.W)
