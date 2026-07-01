import torch

from src.manifolds.isometrized_euclidean import l2IsometrizedEuclidean

class l2IsometrizedVectorEuclidean(l2IsometrizedEuclidean):
    def __init__(self, vector_euclidean, num_intervals=10):
        super().__init__(vector_euclidean, num_intervals=num_intervals)

    def l2_inner(self, X, Y):
        """

        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        return torch.einsum("NMi,NLi->NML", X, Y)
    
    def l2_norm(self, X):
        """

        :param X: N x M x d
        :return: N x M
        """
        return torch.einsum("NMi,NMi->NM", X, X).sqrt()
    
    def monotonicity_upper_bound(self, x, X, y, Y):
        """

        :param x: N x M x d
        :param X: N x M x d
        :param y: N x L x d
        :param Y: N x L x d
        :return: N x M x L
        """
        N, M, d = x.shape[0], x.shape[1], x.shape[2]
        L = y.shape[1]

        log_x_y = self.log(x, y)  # N x M x L
        
        P_y_x_X = self.parallel_transport(x[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, d), 
                                          X[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, 1, d), 
                                          y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, 1, d)
                                          ).reshape(N, M, L, d) # N x M x L x d

        P_y_x_log_x_y = self.parallel_transport(x[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, d),  
                                                log_x_y.reshape(N * M * L, 1, 1, d), 
                                                y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, 1, d)
                                                ).reshape(N, M, L, d) # N x M x L x d

        dist_x_y = self.distance(x, y)  # N x M x L

        return self.inner(y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, d), 
                          (Y[:,None] - P_y_x_X).reshape(N * M * L, 1, d),
                          P_y_x_log_x_y.reshape(N * M * L, 1, d)
                          ).reshape(N, M, L) / dist_x_y**2 # N x M x L
    
    def lipschitz_lower_bound(self, x, X, y, Y):
        """

        :param x: N x M x d
        :param X: N x M x d
        :param y: N x L x d
        :param Y: N x L x d
        :return: N x M x L
        """
        N, M, d = x.shape[0], x.shape[1], x.shape[2]
        L = y.shape[1]

        P_y_x_X = self.parallel_transport(x[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, d), 
                                          X[:,:,None].repeat(1, 1, L, 1).reshape(N * M * L, 1, 1, d), 
                                          y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, 1, d)
                                          ).reshape(N, M, L, d) # N x M x L x d

        dist_x_y = self.distance(x, y)  # N x M x L

        return self.norm(y[:,None].repeat(1, M, 1, 1).reshape(N * M * L, d), 
                          (Y[:,None] - P_y_x_X).reshape(N * M * L, 1, d)
                          ).reshape(N, M, L) / dist_x_y # N x M x L