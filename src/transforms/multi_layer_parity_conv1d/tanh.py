from src.transforms.multi_layer_parity_conv1d import MultiLayerParityConv1DTransform
from src.nn.module.activation.tanh import TanhActivation

class MultiLayerTanhParityConv1DTransform(MultiLayerParityConv1DTransform):
    def __init__(self, in_channels, length, kernel_size, latent_channels, order=2, parity=0):
        super().__init__(in_channels, length, kernel_size, latent_channels, TanhActivation, activation_args={'order':order}, parity=parity)
        