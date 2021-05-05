from torch import nn
from torch.nn.functional import interpolate

__all__ = ["Transformer"]


class Interpolate(nn.Module):
    def __init__(self, scale_factor=1.0, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        return interpolate(input, scale_factor=self.scale_factor, mode=self.mode)

    def extra_repr(self):
        extras = []
        if self.scale_factor:
            extras.append(f"scale_factor={self.scale_factor}")
        if self.mode != "nearest":
            extras.append(f"mode={self.mode}")
        return ", ".join(extras)


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        upsample=False,
        norm=True,
        activation=True,
    ):
        super().__init__()
        self.upsample = Interpolate(scale_factor=stride) if upsample else None
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1 if upsample else stride
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if norm else None
        self.activation = nn.ReLU() if activation else None

    def forward(self, input):
        if self.upsample:
            input = self.upsample(input)

        output = self.conv(self.pad(input))

        if self.norm:
            output = self.norm(output)
        if self.activation:
            output = self.activation(output)

        return output


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv(channels, channels, kernel_size=3)
        self.conv2 = Conv(channels, channels, kernel_size=3, activation=False)

    def forward(self, input):
        output = self.conv2(self.conv1(input))
        return output + input


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            Conv(3, 32, kernel_size=9),
            Conv(32, 64, kernel_size=3, stride=2),
            Conv(64, 128, kernel_size=3, stride=2),
            Residual(128),
            Residual(128),
            Residual(128),
            Residual(128),
            Residual(128),
        )
        self.decoder = nn.Sequential(
            Conv(128, 64, kernel_size=3, stride=2, upsample=True),
            Conv(64, 32, kernel_size=3, stride=2, upsample=True),
            Conv(32, 3, kernel_size=9, norm=False, activation=False),
        )

    def forward(self, input):
        return self.decoder(self.encoder(input))