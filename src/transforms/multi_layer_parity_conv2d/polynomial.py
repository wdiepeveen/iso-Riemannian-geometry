from src.transforms.multi_layer_parity_conv2d import MultiLayerParityConv2DTransform
from src.nn.module.activation.polynomial import PolynomialActivation

class MultiLayerPolynomialParityConv2DTransform(MultiLayerParityConv2DTransform):
    def __init__(self, in_channels, height, width, kernel_size, latent_channels, order=2, parity=0):
        super().__init__(in_channels, height, width, kernel_size, latent_channels, PolynomialActivation, activation_args={'order':order}, parity=parity)
        