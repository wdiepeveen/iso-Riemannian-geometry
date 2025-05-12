from src.transforms.parity import ParityTransform
from src.nn.module.activation.polynomial import PolynomialActivation

class PolynomialParityTransform(ParityTransform):
    def __init__(self, features, order=2, parity=0):
        super().__init__(features, PolynomialActivation, activation_args={'order':order}, parity=parity)
