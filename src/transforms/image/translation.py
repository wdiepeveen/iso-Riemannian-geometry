from src.transforms.image import ImageTransform

class TranslationImageTransform(ImageTransform):
    def __init__(self, translation):
        super().__init__(translation.shape[0], translation.shape[1], translation.shape[2])
        self.translation = translation

    def forward(self, inputs, context=None):
        return inputs - self.translation[None].to(inputs.device), 0.0

    def inverse(self, inputs, context=None):
        return inputs + self.translation[None].to(inputs.device), 0.0