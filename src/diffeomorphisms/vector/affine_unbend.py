import torch
from torch.autograd.functional import jvp

from src.diffeomorphisms.vector import VectorDiffeomorphism

import torch
from src.diffeomorphisms.vector import VectorDiffeomorphism


class AffineUnbendVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, angle, eta) -> None:
        super().__init__(2)

        angle = torch.as_tensor(angle)
        c = torch.cos(angle)
        s = torch.sin(angle)

        self.rot = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c]),
        ])
        self.eta = torch.as_tensor(eta)

    def forward(self, x):
        z = torch.einsum("ab,Nb->Na", self.rot, x)
        z0 = z[:, 0] - z[:, 1] ** 2 / 6
        z1 = torch.asinh(self.eta * z[:, 1])
        return torch.stack((z0, z1), dim=1)

    def inverse(self, y):
        z1 = torch.sinh(y[:, 1]) / self.eta
        z0 = y[:, 0] + z1 ** 2 / 6
        z = torch.stack((z0, z1), dim=1)
        return torch.einsum("ab,Nb->Na", self.rot.T, z)

    def differential_forward(self, x, X):
        z = torch.einsum("ab,Nb->Na", self.rot, x)
        W = torch.einsum("ab,Nb->Na", self.rot, X)

        Y0 = W[:, 0] - (z[:, 1] / 3.0) * W[:, 1]
        Y1 = self.eta * W[:, 1] / torch.sqrt(1.0 + (self.eta * z[:, 1]) ** 2)
        return torch.stack((Y0, Y1), dim=1)

    def differential_inverse(self, y, Y):
        z1 = torch.sinh(y[:, 1]) / self.eta
        Z1 = torch.cosh(y[:, 1]) * Y[:, 1] / self.eta
        Z0 = Y[:, 0] + (z1 / 3.0) * Z1
        Z = torch.stack((Z0, Z1), dim=1)
        return torch.einsum("ab,Nb->Na", self.rot.T, Z)

# class AffineUnbendVectorDiffeomorphism(VectorDiffeomorphism):
#     def __init__(self, angle, eta) -> None:
#         super().__init__(2)
#         self.rot = torch.tensor([
#             [torch.cos(torch.tensor([angle])), -torch.sin(torch.tensor([angle]))], 
#             [torch.sin(torch.tensor([angle])), torch.cos(torch.tensor([angle]))]
#             ])
#         self.eta = eta

#     def forward(self, x):
#         y = x.clone()
#         y = torch.einsum("ab,Nb->Na", self.rot, y)
#         y[:, 0] = y[:, 0] - y[:, 1] ** 2 / 6
#         y[:, 1] = torch.asinh(self.eta * y[:, 1])
#         return y

#     def inverse(self, y):
#         x = y.clone()
#         x[:, 0] = y[:, 0] + (torch.sinh(y[:, 1]) / self.eta) ** 2 / 6
#         x[:, 1] = torch.sinh(y[:, 1]) / self.eta
#         x = torch.einsum("ab,Na->Nb", self.rot, x)
#         return x

#     def differential_forward(self, x, X):
#         z = torch.einsum("ab,Nb->Na", self.rot, x)
#         W = torch.einsum("ab,Nb->Na", self.rot, X)

#         Y = X.clone()
#         Y[:, 0] = W[:, 0] - (z[:, 1] / 3.0) * W[:, 1]
#         Y[:, 1] = self.eta * W[:, 1] / torch.sqrt(1.0 + (self.eta * z[:, 1]) ** 2)
#         return Y

#     def differential_inverse(self, y, Y):
#         z1 = torch.sinh(y[:, 1]) / self.eta

#         Z = Y.clone()
#         Z[:, 1] = torch.cosh(y[:, 1]) * Y[:, 1] / self.eta
#         Z[:, 0] = Y[:, 0] + (z1 / 3.0) * Z[:, 1]

#         X = torch.einsum("ab,Nb->Na", self.rot.T, Z)
#         return X
