import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros', device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.mask = self.generate_weight_mask()

    def forward(self, x):
        weight = self.weight * self.mask.to(x.device)
        return nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def generate_weight_mask(self):
        mask = torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                mask[:, :, i, j] = 0 if (i + j) % 2 == 0 else 1
        return mask