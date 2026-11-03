import torch

from src.transforms.image import ImageTransform

class DiagonalLinearImageTransform(ImageTransform):
    def __init__(self, diagonal):
        super().__init__(diagonal.shape[0], diagonal.shape[1], diagonal.shape[2])
        self.diagonal = diagonal

    def forward(self, x, context=None):
        z = x * self.diagonal[None].to(x.device)  # Element-wise multiplication for diagonal transformation
        log_abs_det = torch.sum(torch.log(torch.abs(self.diagonal))).to(x.device)  # Log determinant is sum of log of absolute values of diagonal entries
        return z, log_abs_det

    def inverse(self, z, context=None):
        x = z / self.diagonal[None].to(z.device)  # Element-wise division for inverse transformation
        log_abs_det = -torch.sum(torch.log(torch.abs(self.diagonal))).to(z.device)  # Inverse transformation negates the log determinant
        return x, log_abs_det