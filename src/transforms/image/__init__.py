from nflows.transforms import Transform

class ImageTransform(Transform):
    def __init__(self, in_channels, height, width):
        super().__init__()
        self.d = in_channels * height * width
        self.C = in_channels
        self.H = height
        self.W = width