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
    
    def geodesic(self, x, y, t, no_grad=True):
        """
        
        :param x: N x M x [Epoint] 
        :param y: N x L x [Epoint] 
        :param t: K
        :return: N x M x L x K x [Epoint] 
        """
        if not no_grad:
            tau = self.tau(x, y, t, no_grad=False)
            return self.euclidean.geodesic(x, y, tau)
        else:
            with torch.no_grad():
                tau = self.tau(x, y, t, no_grad=False)
                return self.euclidean.geodesic(x, y, tau)

    def log(self, x, y, no_grad=True): 
        """
        
        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :return: N x M x L x [Evector] 
        """
        if not no_grad:
            logs = self.euclidean.log(x, y)
            iso_dists = self.distance(x, y, no_grad=False)  # shape: N x M x L
            norm_axes = list(range(3, logs.dim()))
            norms = logs.norm(2, norm_axes)  # shape: N x M x L
            for _ in norm_axes:
                norms = norms.unsqueeze(-1)
                iso_dists = iso_dists.unsqueeze(-1)
            return (iso_dists / norms) * logs
        else:
            with torch.no_grad():
                return self.log(x, y, no_grad=False)
        
    def exp(self, x, X, no_grad=True):
        """
        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M x [Epoint]
        """
        if not no_grad:
            N, M = X.shape[:2]
            for i in range(self.num_intervals):
                if i == 0:
                    c = self.euclidean.exp(x, X / self.num_intervals)
                    log_x_c = X / self.num_intervals
                else:
                    log_c_x = self.euclidean.log(c, x[:,None])[:,:,0]
                    log_c_x_norm = log_c_x.norm(2, [i for i in range(2, len(log_c_x.shape))]) # N x M
                    log_x_c_norm = log_x_c.norm(2, [i for i in range(2, len(log_x_c.shape))]) # N x M
                    log_x_c = log_x_c + (log_x_c_norm / log_c_x_norm).reshape(N, M, *[1 for _ in range(2, len(log_x_c.shape))]) * X / self.num_intervals
                    c = self.euclidean.exp(x, log_x_c)
                    # transported = self.parallel_transport(
                    #     x.reshape(N, 1, *x.shape[1:]),
                    #     X.reshape(N, 1, M, 1, *X.shape[2:]),
                    #     c.reshape(N, M, *x.shape[1:]),
                    #     no_grad=False
                    # )[:, 0, :, 0].reshape(X.shape).clone() # N x M x [Evector]

                    # c = self.euclidean.exp(
                    #     c.reshape(N * M, *x.shape[1:]).clone(),
                    #     (transported.reshape(N * M, 1, *X.shape[2:]) / self.num_intervals).clone()
                    # ).reshape(X.shape).clone()

            return c
        else:
            with torch.no_grad():
                return self.exp(x, X, no_grad=False)
        
    def distance(self, x, y, no_grad=True): 
        """
        Summed segment length of discrete geodesic approximation
        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :return: N x M x L
        """
        if not no_grad:
            geos = self.euclidean.geodesic(x, y, torch.linspace(0., 1., self.num_intervals + 1, device=x.device))
            return ((geos[:,:,:,1:] - geos[:,:,:,:-1])**2).sum([i for i in range(4, len(geos.shape))]).sqrt().sum(-1)
        else:
            with torch.no_grad():
                return self.distance(x, y, no_grad=False)
    
    def parallel_transport(self, x, X, y, no_grad=True):
        """

        :param x: N x M x [Epoint]
        :param X: N x M x L x K x [Evector]
        :param y: N x L x [Epoint]
        :return: N x M x L x K x [Evector]
        """
        if not no_grad:
            log_x_y = self.euclidean.log(x, y) # N x M x L x [Evector]
            log_x_y_norm = log_x_y.norm(2, [i for i in range(3, len(log_x_y.shape))]) # N x M x L
            log_y_x = self.euclidean.log(y, x).transpose(1,2) # N x M x L x [Evector]
            log_y_x_norm = log_y_x.norm(2, [i for i in range(3, len(log_y_x.shape))]) # N x M x L

            prefactor = (log_x_y_norm / log_y_x_norm).reshape(x.shape[0], x.shape[1], y.shape[1], *[1 for _ in range(3, len(X.shape))])
            return prefactor * self.euclidean.parallel_transport(x, X, y)
        else:
            with torch.no_grad():
                return self.parallel_transport(x, X, y, no_grad=False)
    
    def tau(self, x, y, t, no_grad=True):
        """

        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :param t: K
        :return: N x M x L x K 
        """
        if not no_grad:
            N, M = x.shape[0:2]
            L = y.shape[1]

            # compute geodesics at discrete points
            geos = self.euclidean.geodesic(x, y, torch.linspace(0., 1., self.num_intervals + 1, device=x.device))
            interval_lengths = ((geos[:,:,:,1:] - geos[:,:,:,:-1])**2).sum([i for i in range(4, len(geos.shape))]).sqrt() # N x M x L x self.num_intervals

            # compute tau at discrete points
            disc_tau = torch.cat([torch.zeros([N,M,L,1], device=x.device), torch.cumsum(interval_lengths,-1) / torch.sum(interval_lengths, -1)[:,:,:,None]], -1) # N x M x L x self.num_intervals+1
            
            # compute tau at continuous points
            return torch.clamp(torch.sum(1 / self.num_intervals * torch.clamp((t[None,None,None,:,None] - disc_tau[:,:,:,None,:-1]) / (disc_tau[:,:,:,None,1:] - disc_tau[:,:,:,None,:-1]), 0., 1.), -1), 0., 1.)
        else:
            with torch.no_grad():
                return self.tau(x, y, t, no_grad=False)
    