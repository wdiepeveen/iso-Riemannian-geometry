import torch

from src.dimension_reduction import DimensionReductionSolver

class l2PGASolver(DimensionReductionSolver):
    """ Implements base class for solving the PGA problem of finding some rank-r matrix Xi_x \in R^{n x d_1 x ... x d_n} such that \|Y - exp_x(Xi_x)\|_F^2 is small """
    def __init__(self, N, d, data, euclidean, base_point) -> None:
        super().__init__(N, d, data)

        self.euclidean = euclidean
        self.base_point = base_point

        with torch.no_grad():
            self.log_x_data = self.euclidean.log(self.base_point, self.data).detach()  # ∈ R^{n x d}

    def solve(self, rank, discard_highest_error=None):
        with torch.no_grad():
            print(f"Computing rank {rank} approximation on tangent space")
            Xi = self.get_Xi(rank)
            print(f"Computing rank {rank} approximation on euclidean space")
            exp_x_Xi = self.euclidean.exp(self.base_point, Xi).detach()
            print(f"Computing rank {rank} errors")
            errors = ((self.data - exp_x_Xi)**2).sum(tuple(range(1, self.data.dim())))

            error = errors.mean()
            data_norm = (self.data**2).mean(0).sum()
            rel_error = error / data_norm
            if discard_highest_error is not None:
                assert discard_highest_error > 0
                _, sorted_indices = torch.sort(errors)
                indices = sorted_indices[:-discard_highest_error]
                discarded_indices = sorted_indices[-discard_highest_error:]
                discarded_data = self.data[discarded_indices]

                restricted_error = errors[indices].mean()
                restricted_data_norm = (self.data[indices]**2).mean(0).sum()
                restricted_rel_error = restricted_error / restricted_data_norm

                return Xi, exp_x_Xi, {'rel_error':rel_error, 'restricted_rel_error':restricted_rel_error, 'discarded_data':discarded_data, 'errors':errors}
            else:
                return Xi, exp_x_Xi, {'rel_error':rel_error}
    
    def get_Xi(self, rank):
        raise NotImplementedError(
            "Subclasses should implement this"
        )
