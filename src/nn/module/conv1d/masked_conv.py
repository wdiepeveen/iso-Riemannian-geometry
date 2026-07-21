import torch
import torch.nn as nn

class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros', device=None, dtype=None):
        padding = (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.mask = self.generate_weight_mask()

    def forward(self, x):
        weight = self.weight * self.mask.to(x.device)
        return nn.functional.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def generate_weight_mask(self):
        mask = torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0])
        for i in range(self.kernel_size[0]):
            mask[:, :, i] = 0 if i % 2 == 0 else 1
        return mask