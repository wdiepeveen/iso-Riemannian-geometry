from src.manifolds.euclidean import Euclidean

class SequenceFromVectorEuclidean(Euclidean):
    def __init__(self, in_channels, length):
        super().__init__(in_channels * length)
        self.C = in_channels
        self.L = length
        self.vector_euclidean = None

    def inner(self, x, X, Y):
        """

        :param x: N x (C x L) 
        :param X: N x M x (C x L)
        :param Y: N x L x (C x L)
        :return: N x M x L
        """
        N = x.shape[0]
        M = X.shape[1]
        L = Y.shape[1]
        return self.vector_euclidean.inner(
            x.reshape(N, self.C * self.L), 
            X.reshape(N, M, self.C * self.L),
            Y.reshape(N, L, self.C * self.L)
            )
    
    def norm(self, x, X):
        """

        :param x: N x (C x L) 
        :param X: N x M x (C x L) 
        :return: N x M
        """
        N = x.shape[0]
        M = X.shape[1]
        return self.vector_euclidean.norm(
            x.reshape(N, self.C * self.L), 
            X.reshape(N, M, self.C * self.L)
            )

    def barycentre(self, x, tol=None, max_iter=None, step_size=None, red_coef=None):
        """

        :param x: N x (C x L)
        :return: (C x L)
        """
        N = x.shape[0]
        return self.vector_euclidean.barycentre(x.reshape(N, self.C * self.L), tol=tol, max_iter=max_iter, step_size=step_size, red_coef=red_coef).reshape(self.C, self.L)
    
    def geodesic(self, x, y, t):
        """

        :param x: N x M x (C x L)
        :param y: N x L x (C x L)
        :param t: K or N x M x L x K
        :return: N x M x L x K x (C x L)
        """
        N, M = x.shape[0:2]
        L = y.shape[1]
        K = t.shape[-1]
        return self.vector_euclidean.geodesic(
                x.reshape(N, M, self.C * self.L),
                y.reshape(N, L, self.C * self.L),
                t
                ).reshape(N, M, L, K, self.C, self.L)
            

    def log(self, x, y):
        """

        :param x: N x M x (C x L)
        :param y: N x L x (C x L)
        :return: N x M x L x (C x L)
        """
        N, M = x.shape[0:2]
        L = y.shape[1]
        return self.vector_euclidean.log(
            x.reshape(N, M, self.C * self.L),
            y.reshape(N, L, self.C * self.L)
            ).reshape(N, M, L, self.C, self.L)

    def exp(self, x, X):
        """

        :param x: N x (C x L)
        :param X: N x M x (C x L)
        :return: N x M x (C x L)
        """
        N, M = X.shape[0:2]
        return self.vector_euclidean.exp(
            x.reshape(N, self.C * self.L),
            X.reshape(N, M, self.C * self.L)
            ).reshape(N, M, self.C, self.L)
    
    def distance(self, x, y):
        """

        :param x: N x M x (C x L)
        :param y: N x L x (C x L)
        :return: N x M x L
        """
        N, M = x.shape[0:2]
        L = y.shape[1]
        return self.vector_euclidean.distance(
            x.reshape(N, M, self.C * self.L),
            y.reshape(N, L, self.C * self.L)
            )

    def parallel_transport(self, x, X, y):
        """

        :param x: N x M x (C x L)
        :param X: N x M x L x K x (C x L)
        :param y: N x L x (C x L)
        :return: N x M x L x K x (C x L)
        """
        N, M = x.shape[0:2]
        L = y.shape[1]
        K = X.shape[2]
        return self.vector_euclidean.parallel_transport(
            x.reshape(N, M, self.C * self.L),
            X.reshape(N, M, L, K, self.C * self.L),
            y.reshape(N, L, self.C * self.L)
            ).reshape(N, M, L, K, self.C, self.L)
    