from src.dimension_reduction.principal_geodesic_analysis import l2PGASolver

class l2PGAVectorSolver(l2PGASolver):
    def __init__(self, data, vector_euclidean, base_point):
        super().__init__(data.shape[0], data.shape[1], data, vector_euclidean, base_point)