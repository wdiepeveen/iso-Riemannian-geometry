
from src.transforms.image import ImageTransform

class ToVecImageTransform(ImageTransform):
    def __init__(self, in_channels, height, width):
        super().__init__(in_channels, height, width)

    def forward(self, x, context=None):
        """
        
        :param x: N x C x H x W
        :return: N x (C*H*W)
        """
        N = x.shape[0]
        logabsdet = x.new_zeros(N)
        return x.reshape(N, -1), logabsdet   # or x.contiguous().view(N, -1)
        # N= x.shape[0]
        # logabsdet = 0.0
        # return x.view(N, -1), logabsdet
    
    def inverse(self, y, context=None):
        """
        :param y: N x (C*H*W)
        :return: N x C x H x W
        """
        N = y.shape[0]
        logabsdet = y.new_zeros(N)
        return y.reshape(N, self.C, self.H, self.W), logabsdet
        # N = y.shape[0]
        # logabsdet = 0.0
        # return y.view(N, self.C, self.H, self.W), logabsdet