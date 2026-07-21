from torch.autograd.functional import jvp, vjp
from src.diffeomorphisms.sequence import SequenceDiffeomorphism

class NFlowSequenceDiffeomorphism(SequenceDiffeomorphism):
    def __init__(self, in_channels, length, sequence_nflow):
        super().__init__(in_channels, length)
        self.nflow = sequence_nflow

    def forward(self, x):
        """
        Forward pass through the diffeomorphism.
        :param x: N x (C, L)
        :return: N x (C, L)
        """
        out, _ = self.nflow._transform(x, context=None)
        return out

    def inverse(self, y):
        """
        Inverse pass through the diffeomorphism.
        :param y: N x (C, L)
        :return: N x (C, L)
        """
        out, _ = self.nflow._transform.inverse(y, context=None)
        return out

    def differential_forward(self, x, X):
        """
        Compute the differential map of phi at x for a vector X.

        :param x: N x (C, L)
        :param X: N x (C, L)
        :return: N x (C, L)
        """
        _, out = jvp(
            lambda x: self.nflow._transform(x, context=None)[0],
            (x,),
            (X,),
            create_graph=True,
            strict=True,
        )
        return out

    def differential_inverse(self, y, Y):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.

        :param y: N x (C, L)
        :param Y: N x (C, L)
        :return: N x (C, L)
        """
        _, out = jvp(
            lambda y: self.nflow._transform.inverse(y, context=None)[0],
            (y,),
            (Y,),
            create_graph=True,
            strict=True,
        )
        return out

    def adjoint_differential_forward(self, x, X, context=None):
        """
        Compute the adjoint differential map of phi at x for a vector X.

        :param x: N x (C, L)
        :param X: N x (C, L)
        :return: N x (C, L)
        """
        _, vjp_result = vjp(
            lambda x: self.nflow._transform(x, context=context)[0],
            (x,),
            (X,),
            create_graph=True,
            strict=True,
        )
        return vjp_result[0]

    def adjoint_differential_inverse(self, y, Y, context=None):
        """
        Compute the adjoint differential map of the inverse of phi at y for a vector Y.

        :param y: N x (C, L)
        :param Y: N x (C, L)
        :return: N x (C, L)
        """
        _, vjp_result = vjp(
            lambda y: self.nflow._transform.inverse(y, context=context)[0],
            (y,),
            (Y,),
            create_graph=True,
            strict=True,
        )
        return vjp_result[0]
    