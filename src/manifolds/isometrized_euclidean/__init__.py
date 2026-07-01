import torch

from src.manifolds import Manifold

class l2IsometrizedEuclidean(Manifold):
    def __init__(self, euclidean, num_intervals=10):
        super().__init__(euclidean.d)

        self.euclidean = euclidean
        self.num_intervals = num_intervals

    def l2_inner(self, X, Y):
        """

        :param X: N x M x [Evector]
        :param Y: N x L x [Evector]
        :return: N x M x L
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def l2_norm(self, X):
        """

        :param X: N x M x [Evector]
        :return: N x M
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def inner(self, x, X, Y):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :param Y: N x L x [Evector]
        :return: N x M x L
        """
        return self.euclidean.inner(x, X, Y)
    
    def norm(self, x, X):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M
        """
        return self.euclidean.norm(x, X)

    def barycentre(self, x, tol=1e-2, max_iter=100, step_size=1/2, red_coef=0.5, iso_exp=False, log_errors=False):
        """

        :param x: N x [Epoint]
        :return: [Epoint]
        """
        k = 1
        with torch.no_grad():
            y = self.euclidean.barycentre(x)
            
            gradient = - torch.mean(self.log(y[None,None], x[None,:])[0,0],0)
            error = self.l2_norm(gradient[None,None])[0,0]
            if log_errors:
                errors = [error.item()]
            while k <= max_iter and error >= tol:
                step_size_ = step_size
                descent = False
                while not descent:
                    if iso_exp:
                        y_tmp = self.exp(y[None], - step_size_ * gradient[None,None])[0,0]
                    else:
                        y_tmp = self.euclidean.exp(y[None], - step_size_ * gradient[None,None])[0,0]
                    gradient_tmp = - torch.mean(self.log(y_tmp[None,None], x[None,:])[0,0],0)
                    error_tmp = self.l2_norm(gradient_tmp[None,None])[0,0]
                    descent = error_tmp < error
                    if not descent:
                        step_size_ *= red_coef
                y = y_tmp
                gradient = gradient_tmp
                error = error_tmp
                if log_errors:
                    errors.append(error.item())
                print(f"iteration {k} | error = {error.item()}")
                k+=1

            print(f"gradient descent was terminated after reaching a error {error.item()} in {k-1} iterations")

            if log_errors:
                return y, errors
            else:
                return y
    
    def geodesic(self, x, y, t):
        """
        
        :param x: N x M x [Epoint] 
        :param y: N x L x [Epoint] 
        :param t: K
        :return: N x M x L x K x [Epoint] 
        """
        with torch.no_grad():
            tau = self.tau(x, y, t)
            return self.euclidean.geodesic(x, y, tau)

    def log(self, x, y): 
        """
        
        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :return: N x M x L x [Evector] 
        """
        with torch.no_grad():
            logs = self.euclidean.log(x, y)
            iso_dists = self.distance(x, y)
            norm_axes = list(range(3, logs.dim()))
            norms = logs.norm(2, norm_axes)  # shape: N x M x L
            for _ in norm_axes:
                norms = norms.unsqueeze(-1)
                iso_dists = iso_dists.unsqueeze(-1)
            return (iso_dists / norms) * logs
        
    def exp(self, x, X):
        """
        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M x [Epoint]
        """
        with torch.no_grad():
            N, M = X.shape[:2]
            x_base = x[:, None].expand(-1, M, *x.shape[1:])  # N x M x [Epoint]
            for i in range(self.num_intervals):
                if i == 0:
                    c = self.euclidean.exp(x, X / self.num_intervals) # N x M x [Epoint]
                else:
                    transported = self.parallel_transport(
                        x_base.reshape(N*M, 1, *x.shape[1:]),           # N*M x 1 x [Epoint]
                        X.reshape(N*M, 1, 1, *X.shape[2:]),             # N*M x 1 x 1 x [Evector]
                        c.reshape(N*M, 1, *x.shape[1:])                 # N*M x 1 x [Epoint]
                    )[:,0,0,0].reshape(X.shape)                         # N x M [Evector]
                    # compute next point along geodesic
                    c = self.euclidean.exp(c.reshape(N*M, *x.shape[1:]), transported.reshape(N*M, 1, *X.shape[2:]) / self.num_intervals).reshape(X.shape) # N x M x [Epoint]

            return c
        
    def distance(self, x, y): 
        """
        Summed segment length of discrete geodesic approximation
        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :return: N x M x L
        """
        with torch.no_grad():
            geos = self.euclidean.geodesic(x, y, torch.linspace(0., 1., self.num_intervals + 1, device=x.device))
            return ((geos[:,:,:,1:] - geos[:,:,:,:-1])**2).sum([i for i in range(4, len(geos.shape))]).sqrt().sum(-1)
    
    def parallel_transport(self, x, X, y):
        """

        :param x: N x M x [Epoint]
        :param X: N x M x K x [Evector]
        :param y: N x L x [Epoint]
        :return: N x M x L x K x [Evector]
        """
        with torch.no_grad():
            log_x_y = self.euclidean.log(x, y) # N x M x L x [Evector]
            log_x_y_norm = log_x_y.norm(2, [i for i in range(3, len(log_x_y.shape))]) # N x M x L
            log_y_x = self.euclidean.log(y, x).transpose(1,2) # N x M x L x [Evector]
            log_y_x_norm = log_y_x.norm(2, [i for i in range(3, len(log_y_x.shape))]) # N x M x L

            prefactor = log_x_y_norm / log_y_x_norm
            return prefactor[:,:,:,None,None] * self.euclidean.parallel_transport(x, X, y)
    
    def tau(self, x, y, t):
        """

        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :param t: K
        :return: N x M x L x K 
        """
        with torch.no_grad():
            N, M = x.shape[0:2]
            L = y.shape[1]

            # compute geodesics at discrete points
            geos = self.euclidean.geodesic(x, y, torch.linspace(0., 1., self.num_intervals + 1, device=x.device))
            interval_lengths = ((geos[:,:,:,1:] - geos[:,:,:,:-1])**2).sum([i for i in range(4, len(geos.shape))]).sqrt() # N x M x L x self.num_intervals

            # compute tau at discrete points
            disc_tau = torch.cat([torch.zeros([N,M,L,1], device=x.device), torch.cumsum(interval_lengths,-1) / torch.sum(interval_lengths, -1)[:,:,:,None]], -1) # N x M x L x self.num_intervals+1
            
            # compute tau at continuous points
            return torch.clamp(torch.sum(1 / self.num_intervals * torch.clamp((t[None,None,None,:,None] - disc_tau[:,:,:,None,:-1]) / (disc_tau[:,:,:,None,1:] - disc_tau[:,:,:,None,:-1]), 0., 1.), -1), 0., 1.)
    