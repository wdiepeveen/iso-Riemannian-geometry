from nflows.transforms import CompositeTransform

class ImageCompositeTransform(CompositeTransform):
    def __init__(self, in_channels, height, width, image_transforms):
        super().__init__(image_transforms)
        self.C = in_channels
        self.H = height
        self.W = width