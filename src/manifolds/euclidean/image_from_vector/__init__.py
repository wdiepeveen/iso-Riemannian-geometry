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

    def barycentre(self, x, tol=None, max_iter=None, step_size=None, red_coef=None):
        """

        :param x: N x (C x H x W)
        :return: (C x H x W)
        """
        N = x.shape[0]
        return self.vector_euclidean.barycentre(x.reshape(N, self.C * self.H * self.W), tol=tol, max_iter=max_iter, step_size=step_size, red_coef=red_coef).reshape(self.C, self.H, self.W)
    
    def geodesic(self, x, y, t):
        """

        :param x: N x M x (C x H x W)
        :param y: N x L x (C x H x W)
        :param t: K or N x M x L x K
        :return: N x M x L x K x (C x H x W)
        """
        N, M = x.shape[0:2]
        L = y.shape[1]
        K = t.shape[-1]
        return self.vector_euclidean.geodesic(
                x.reshape(N, M, self.C * self.H * self.W),
                y.reshape(N, L, self.C * self.H * self.W),
                t
                ).reshape(N, M, L, K, self.C, self.H, self.W)
            

    def log(self, x, y):
        """

        :param x: N x M x (C x H x W)
        :param y: N x L x (C x H x W)
        :return: N x M x L x (C x H x W)
        """
        N, M = x.shape[0:2]
        L = y.shape[1]
        return self.vector_euclidean.log(
            x.reshape(N, M, self.C * self.H * self.W),
            y.reshape(N, L, self.C * self.H * self.W)
            ).reshape(N, M, L, self.C, self.H, self.W)

    def exp(self, x, X):
        """

        :param x: N x (C x H x W)
        :param X: N x M x (C x H x W)
        :return: N x M x (C x H x W)
        """
        N, M = X.shape[0:2]
        return self.vector_euclidean.exp(
            x.reshape(N, self.C * self.H * self.W),
            X.reshape(N, M, self.C * self.H * self.W)
            ).reshape(N, M, self.C, self.H, self.W)
    
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

        :param x: N x M x (C x H x W)
        :param X: N x M x K x (C x H x W)
        :param y: N x L x (C x H x W)
        :return: N x M x L x K x (C x H x W)
        """
        N, M = x.shape[0:2]
        L = y.shape[1]
        K = X.shape[2]
        return self.vector_euclidean.parallel_transport(
            x.reshape(N, M, self.C * self.H * self.W),
            X.reshape(N, M, K, self.C * self.H * self.W),
            y.reshape(N, L, self.C * self.H * self.W)
            ).reshape(N, N, L, K, self.C, self.H, self.W)
    