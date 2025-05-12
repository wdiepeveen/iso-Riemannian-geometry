import torch.nn as nn

class Activation(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.d = num_features
    
    def forward(self, x):
        raise NotImplementedError(
            "Subclasses should implement this"
        )