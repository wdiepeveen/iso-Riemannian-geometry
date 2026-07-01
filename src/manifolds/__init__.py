class Manifold:
    """ Base class describing a manifold of dimension d """

    def __init__(self, d):
        self.d = d
    
    def inner(self, x, X, Y):
        """

        :param x: N x [Mpoint]
        :param X: N x M x [Mvector]
        :param Y: N x L x [Mvector]
        :return: N x M x L
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def norm(self, x, X):
        """

        :param x: N x [Mpoint]
        :param X: N x M x [Mvector]
        :return: N x M
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
        
    def barycentre(self, x, tol=None, max_iter=None, step_size=None, red_coef=None):
        """

        :param x: N x [Mpoint]
        :return: [Mpoint]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def geodesic(self, x, y, t):
        """

        :param x: N x M x [Mpoint]
        :param y: N x L x [Mpoint]
        :param t: K
        :return: N x M x L x K x [Mpoint]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def log(self, x, y):
        """

        :param x: N x M x [Mpoint]
        :param y: N x L x [Mpoint]
        :return: N x M x L x [Mvector]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def exp(self, x, X):
        """

        :param x: N x [Mpoint]
        :param X: N x M x [Mvector]
        :return: N x M x [Mpoint]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def distance(self, x, y):
        """

        :param x: N x M x [Mpoint]
        :param y: N x L x [Mpoint]
        :return: N x M x L
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def parallel_transport(self, x, X, y):
        """

        :param x: N x M x [Mpoint]
        :param X: N x M x K x [Mvector]
        :param y: N x L x [Mpoint]
        :return: N x M x L x K [Mvector]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def manifold_dimension(self):
        return self.d
    