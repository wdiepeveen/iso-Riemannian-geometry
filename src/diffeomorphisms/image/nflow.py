from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.image import ImageDiffeomorphism
        
class NFlowImageDiffeomorphism(ImageDiffeomorphism):
    def __init__(self, in_channels, height, width, image_nflow):
        super().__init__(in_channels, height, width)

        self.nflow = image_nflow

    def forward(self, x):
        """
        Forward pass through the diffeomorphism.
        :param x: N x (C, H, W)
        :return: N x (C, H, W)
        """
        out, _ = self.nflow._transform(x, context=None)
        return out

    def inverse(self, y):
        """
        Inverse pass through the diffeomorphism.
        :param y: N x (C, H, W)
        :return: N x (C, H, W)
        """
        out, _ = self.nflow._transform.inverse(y, context=None)
        return out

    def differential_forward(self, x, X, flattened_out=False):
        """
        Compute the differential map of phi at x for a vector X.
        
        :param x: N x (C, H, W)
        :param X: N x (C, H, W)
        :return: N x (C, H, W)
        """
        _, out = jvp(lambda x: self.nflow._transform(x, context=None)[0], (x,), (X,))
        return out

    def differential_inverse(self, y, Y, flattened_out=False):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.
        
        :param y: N x (C, H, W)
        :param Y: N x (C, H, W)
        :return: N x (C, H, W)
        """
        _, out = jvp(lambda y: self.nflow._transform.inverse(y, context=None)[0], (y,), (Y,))
        return out
    
    def adjoint_differential_forward(self, x, X, context=None):
        """
        Compute the adjoint differential map of phi at x for a vector X.
        
        :param x: N x (C, H, W)
        :param X: N x (C, H, W)
        :return: N x (C, H, W)
        """
        _, vjp_result = vjp(lambda x: self.nflow._transform(x, context=context)[0], x, X)
        return vjp_result[0]

    def adjoint_differential_inverse(self, y, Y, context=None):
        """
        Compute the adjoint differential map of the inverse of phi at y for a vector Y.
        
        :param y: N x (C, H, W)
        :param Y: N x (C, H, W)
        :return: N x (C, H, W)
        """
        _, vjp_result = vjp(lambda y: self.nflow._transform.inverse(y, context=context)[0], (y,), (Y,))
        return vjp_result[0]
    
