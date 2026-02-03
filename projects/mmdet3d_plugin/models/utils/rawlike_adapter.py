import torch
from torch import nn


class RawLikeToSRGBAdapter(nn.Module):
    """A lightweight adapter to map RAW-like inputs back to sRGB distribution."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        init_identity=True,
        use_gamma=True,
        gamma_min=0.1,
        gamma_max=5.0,
        clamp_min=1e-6,
        clamp_max=1.0,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.use_gamma = use_gamma
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(out_channels))
        else:
            self.register_parameter("gamma", None)

        if init_identity and in_channels == out_channels:
            nn.init.eye_(self.proj.weight.data.view(out_channels, -1))
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        x = self.proj(x)
        if self.use_gamma:
            gamma = torch.clamp(self.gamma, min=self.gamma_min, max=self.gamma_max)
            x = torch.clamp(x, min=self.clamp_min)
            x = torch.pow(x, gamma.view(1, -1, 1, 1))
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x
