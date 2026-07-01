import torch

from src.manifolds.euclidean import Euclidean

class VectorEuclidean(Euclidean):
    def __init__(self, d):
        super().__init__(d)

    def monotonicity_upper_bound(self, x, X, y, Y):
        """

        :param x: N x M x d
        :param X: N x M x d
        :param y: N x L x d
        :param Y: N x L x d
        :return: N x M x L
        """
        N, M = x.shape[0], x.shape[1]
        L = y.shape[1]

        log_x_y = self.log(x, y)  # N x M x L

        P_y_x_X = self.parallel_transport(x[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, self.d), 
                                          X[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, 1, self.d), 
                                          y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, 1, self.d)
                                          ).reshape(N, M, L, self.d) # N x M x L x d

        P_y_x_log_x_y = self.parallel_transport(x[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, self.d),  
                                                log_x_y.reshape(N * M * L, 1, 1, self.d), 
                                                y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, 1, self.d)
                                                ).reshape(N, M, L, self.d) # N x M x L x d

        dist_x_y = self.distance(x, y)  # N x M x L

        return self.inner(y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, self.d), 
                          (Y[:,None] - P_y_x_X).reshape(N * M * L, 1, self.d),
                          P_y_x_log_x_y.reshape(N * M * L, 1, self.d)
                          ).reshape(N, M, L) / dist_x_y**2 # N x M x L
    
    def lipschitz_lower_bound(self, x, X, y, Y):
        """

        :param x: N x M x d
        :param X: N x M x d
        :param y: N x L x d
        :param Y: N x L x d
        :return: N x M x L
        """
        N, M = x.shape[0], x.shape[1]
        L = y.shape[1]

        P_y_x_X = self.parallel_transport(x[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, self.d), 
                                          X[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, 1, self.d), 
                                          y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, 1, self.d)
                                          ).reshape(N, M, L, self.d) # N x M x L x d

        dist_x_y = self.distance(x, y)  # N x M x L

        return self.norm(y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, self.d), 
                          (Y[:,None] - P_y_x_X).reshape(N * M * L, 1, self.d)
                          ).reshape(N, M, L) / dist_x_y # N x M x L