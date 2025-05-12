import torch
from torch.autograd.functional import jvp

from src.diffeomorphisms.vector import VectorDiffeomorphism

class AffineUnbendVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, angle, delta, eta) -> None:
        super().__init__(2)
        self.rot = torch.tensor([
            [torch.cos(torch.tensor([angle])), -torch.sin(torch.tensor([angle]))], 
            [torch.sin(torch.tensor([angle])), torch.cos(torch.tensor([angle]))]
            ])
        self.delta = delta
        self.eta = eta

    def huber_loss(self, p):
        """
        Computes the huber loss.
        :param p: Tensor of shape (N,), where N is the batch size.
        :return: Transformed tensor of shape (N,).
        """
        return torch.abs(p) * (p.abs() > self.delta) + (1/(2 * self.delta) * p **2 + 1/2 * self.delta) * (p.abs() <= self.delta)
    
    def derivative_huber_loss(self, p):
        """
        Computes the derivative of the huber loss.
        :param p: Tensor of shape (N,), where N is the batch size.
        :return: Transformed tensor of shape (N,).
        """
        return torch.sign(p) * (p.abs() > self.delta) + (1/self.delta * p) * (p.abs() <= self.delta)

    def forward(self, x):
        """
        Computes the forward transformation of the diffeomorphism.
        :param x: Tensor of shape (N, 2), where N is the batch size.
        :return: Transformed tensor of shape (N, 2).
        """
        y = x.clone()
        y = torch.einsum("ab,Nb->Na", self.rot, y)
        y[:, 0] = y[:, 0] - self.huber_loss(y[:, 1])
        y[:, 1] = torch.tanh(self.eta * y[:, 1])
        return y

    def inverse(self, y):
        """
        Computes the inverse transformation of the diffeomorphism.
        :param y: Tensor of shape (N, 2), where N is the batch size.
        :return: Inverted tensor of shape (N, 2).
        """
        x = y.clone()
        x[:, 0] = y[:, 0] + self.huber_loss(torch.arctanh(y[:, 1]) / self.eta)
        x[:, 1] = torch.arctanh(y[:, 1]) / self.eta
        x = torch.einsum("ab,Na->Nb", self.rot, x)
        return x

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, 2), inputs.
        :param X: Tensor of shape (N, 2), differentials to transform.
        :return: Transformed differential tensor of shape (N, 2).
        """
        # Rotate input vectors
        rotated_x = torch.einsum("ab,Nb->Na", self.rot, x)
        rotated_X = torch.einsum("ab,Nb->Na", self.rot, X)
        
        # Compute nonlinear derivatives
        rx1 = rotated_x[:, 1]
        d_huber = self.derivative_huber_loss(rx1)
        tanh_deriv = self.eta * (1 - torch.tanh(self.eta * rx1).square())
        
        # Apply Jacobian components
        transformed_X0 = rotated_X[:, 0] - rotated_X[:, 1] * d_huber
        transformed_X1 = rotated_X[:, 1] * tanh_deriv
        
        return torch.stack([transformed_X0, transformed_X1], dim=1)

    def differential_inverse(self, y, Y):
        """
        Computes the differential of the inverse transformation.
        :param y: Tensor of shape (N, 2), inputs.
        :param Y: Tensor of shape (N, 2), differentials to invert.
        :return: Inverted differential tensor of shape (N, 2).
        """
        # Compute inverse parameters
        y1 = y[:, 1].clamp(min=-1+1e-6, max=1-1e-6)  # Numerical stability
        x1 = torch.arctanh(y1) / self.eta
        
        # Compute nonlinear derivatives
        d_huber = self.derivative_huber_loss(x1)
        inv_deriv = 1 / (self.eta * (1 - y1.square()))
        
        # Apply inverse Jacobian components
        transformed_Y0 = Y[:, 0] + Y[:, 1] * d_huber * inv_deriv
        transformed_Y1 = Y[:, 1] * inv_deriv
        
        # Rotate back with inverse transformation
        return torch.einsum("ab,Nb->Na", self.rot.T, 
                        torch.stack([transformed_Y0, transformed_Y1], dim=1))
            

