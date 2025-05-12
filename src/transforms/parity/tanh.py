from src.transforms.parity import ParityTransform
from src.nn.module.activation.tanh import TanhActivation

class TanhParityTransform(ParityTransform):
    def __init__(self, features, order=2, parity=0):
        super().__init__(features, TanhActivation, activation_args={'order':order}, parity=parity)
