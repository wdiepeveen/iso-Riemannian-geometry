import torch
import torch.nn as nn
from nflows.transforms import Transform
import nflows.utils.typechecks as check


class ActNorm(Transform):
    def __init__(self, features, eps=1e-6):
        """
        Activation normalization for inputs shaped as:
            - 2D: (B, C)
            - 3D: (B, C, L)
            - 4D: (B, C, H, W)

        Normalization is always per-channel.

        Reference:
        > D. Kingma et al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.features = features
        self.eps = eps

        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def _check_inputs(self, inputs):
        if inputs.dim() not in [2, 3, 4]:
            raise ValueError("Expecting inputs to be a 2D, 3D, or 4D tensor.")
        if inputs.shape[1] != self.features:
            raise ValueError(
                f"Expected {self.features} channels/features, got {inputs.shape[1]}."
            )

    def _broadcastable_scale_shift(self, inputs):
        shape = [1, -1] + [1] * (inputs.dim() - 2)
        return self.scale.view(*shape), self.shift.view(*shape)

    def _num_positions(self, inputs):
        # Number of spatial/temporal positions per channel.
        if inputs.dim() == 2:
            return 1
        return int(torch.tensor(inputs.shape[2:]).prod().item())

    def forward(self, inputs, context=None):
        self._check_inputs(inputs)

        if self.training and not self.initialized:
            self._initialize(inputs)

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = scale * inputs + shift

        batch_size = inputs.shape[0]
        num_positions = self._num_positions(inputs)
        logabsdet = num_positions * torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        self._check_inputs(inputs)

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = (inputs - shift) / scale

        batch_size = inputs.shape[0]
        num_positions = self._num_positions(inputs)
        logabsdet = -num_positions * torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def _initialize(self, inputs):
        """
        Data-dependent initialization so that post-actnorm activations have
        approximately zero mean and unit variance per channel.
        """
        with torch.no_grad():
            if inputs.dim() == 2:
                flat = inputs
            else:
                num_channels = inputs.shape[1]
                flat = inputs.transpose(1, -1).reshape(-1, num_channels)

            std = flat.std(dim=0).clamp_min(self.eps)
            mean = flat.mean(dim=0)

            self.log_scale.data = -torch.log(std)
            self.shift.data = -mean / std
            self.initialized.data.fill_(True)