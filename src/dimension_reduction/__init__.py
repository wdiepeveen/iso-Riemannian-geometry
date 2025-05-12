class DimensionReductionSolver:
    """ Base class for dimension reduction of d-dimensional data Y = (Y_1, ..., Y_n)^T \in R^{n x d} """
    def __init__(self, N, d, data) -> None:
        self.N = N
        self.d = d
        self.data = data

    def solve(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )