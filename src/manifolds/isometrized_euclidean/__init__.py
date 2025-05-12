import torch

from src.curves.boundary_value.discrete import DiscreteBoundaryValueCurve
from src.manifolds import Manifold
from src.time_flows.piecewise_linear import PiecewiseLinearTimeFlow

class l2IsometrizedEuclidean(Manifold):
    def __init__(self, euclidean, num_intervals=100):
        super().__init__(euclidean.d)

        self.euclidean = euclidean
        self.num_intervals = num_intervals

    def barycentre(self, x, tol=1e-1, max_iter=100, step_size=1/4): 
        """

        :param x: N x [Epoint]
        :return: [Epoint]
        """
        k = 1
        y = torch.mean(x,0)
        
        gradient_0 = torch.mean(self.log(y, x),0)
        error = self.norm(y[None], gradient_0[None,None]) + 1e-2
        rel_error = 1.
        while k <= max_iter and rel_error >= tol:
            gradient = torch.mean(self.log(y, x),0)
            y = y + step_size * gradient
            rel_error = self.norm(y[None], gradient[None,None]) / error
            print(f"iteration {k} | rel_error = {rel_error.item()}")
            k+=1

        print(f"gradient descent was terminated after reaching a relative error {rel_error.item()} in {k-1} iterations")

        return y
    
    def geodesic(self, x, y, t):
        """
        Discrete geodesic approximation
        :param x: [Epoint] 
        :param y: [Epoint] 
        :param t: N
        :return: N x [Epoint] 
        """
        z = self.euclidean.geodesic(x,y,torch.linspace(0.,1.,self.num_intervals+1, device=x.device))
        time_flow = PiecewiseLinearTimeFlow(z)
        tau = time_flow(t)
        return self.euclidean.geodesic(x,y,tau)

    def log(self, x, y):
        """
        Discrete geodesic approximation
        :param x: [Epoint]
        :param y: N x [Epoint]
        :return: N x [Evector] 
        """
        N = y.shape[0] 
        logs = torch.zeros_like(y)
        for i in range(N):
            z = self.euclidean.geodesic(x,y[i],torch.linspace(0.,1.,self.num_intervals+1, device=x.device))
            time_flow = PiecewiseLinearTimeFlow(z)
            logs[i] = time_flow.differential_forward(torch.zeros(1, device=x.device)) * self.euclidean.log(x,y[i][None])[0]
            
        return logs

    def exp(self, x, X):
        """
        Discrete geodesic approximation
        :param x: [Epoint]
        :param X: N x [Evector]
        :return: N x [Epoint]
        """
        N = X.shape[0] 
        z_i = torch.cat([x[None] for _ in range(N)],dim=0)
        z_i_ = self.euclidean.exp(x, 1/self.num_intervals * X)

        path_lengths = self.norm(z_i, z_i_[:,None] - z_i[:,None])[:,0]
        prev_path_lengths = torch.zeros_like(path_lengths)
        X_norms = self.norm(z_i, X[:,None])[:,0]

        mask = path_lengths < X_norms

        while mask.any():
            y_i = self.euclidean.geodesic(z_i[mask], z_i_[mask], torch.tensor([2.], device=x.device))
            z_i[mask] = z_i_[mask]
            z_i_[mask] = y_i

            prev_path_lengths[mask] = path_lengths[mask]
            path_lengths[mask] += self.norm(z_i[mask], z_i_[mask][:,None] - z_i[mask][:,None])[:,0]

            mask = path_lengths < X_norms

        t_i = (X_norms - prev_path_lengths) / (path_lengths - prev_path_lengths)
        return torch.cat([((1 - t_i[i]) * z_i[i] + t_i[i] * z_i_[i])[None] for i in range(N)], dim=0)
    
    def distance(self, x, y):
        """
        Summed segment length of discrete geodesic approximation
        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :return: N x M x L
        """
        N, M = x.shape[0:2] 
        N, L = y.shape.shape[0:2]
        distances = torch.zeros(N,M,L)
        for i in range(N):
            for j in range(M):
                for k in range(L):
                    geodesic = DiscreteBoundaryValueCurve(x[i,j], y[i,k], self.num_intervals)
                    geodesic.coefficients = torch.nn.Parameter(self.euclidean.geodesic(x[i,j],y[i,k],torch.linspace(0.,1.,self.num_intervals+1, device=x.device))[1:-1])
        
                    geodesic_t = geodesic(torch.linspace(0.,1.,200, device=x.device))
                    distances[i,j,k] = torch.sum(self.norm(geodesic_t[0:-1], (geodesic_t[1:] - geodesic_t[0:-1])[:,None]))
        return distances