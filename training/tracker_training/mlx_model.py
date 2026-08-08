"""Optional NHWC MLX mirror of HC-DS31.

Importing this module remains safe when MLX is unavailable; constructing the
model then raises an actionable ImportError.
"""

from __future__ import annotations

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # Covered on non-Apple CI through the placeholder class.
    mx = None
    nn = None


if nn is not None:
    class ConvBNActMLX(nn.Module):
        def __init__(self, ci, co, kernel, *, stride=1, groups=1, activate=True):
            super().__init__()
            self.conv = nn.Conv2d(ci, co, kernel, stride=stride,
                                  padding=kernel // 2, groups=groups, bias=False)
            self.bn = nn.BatchNorm(co)
            self.activate = activate

        def __call__(self, x):
            x = self.bn(self.conv(x))
            return nn.relu(x) if self.activate else x


    class DownBlockMLX(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.dw = ConvBNActMLX(ci, ci, 3, stride=2, groups=ci)
            self.pw = ConvBNActMLX(ci, co, 1)

        def __call__(self, x):
            return self.pw(self.dw(x))


    class ResidualBlockMLX(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.dw = ConvBNActMLX(channels, channels, 3, groups=channels)
            self.pw = ConvBNActMLX(channels, channels, 1, activate=False)

        def __call__(self, x):
            return nn.relu(x + self.pw(self.dw(x)))


    class RefineBlockMLX(nn.Module):
        def __init__(self, channels=32):
            super().__init__()
            self.dw = ConvBNActMLX(channels, channels, 3, groups=channels)
            self.pw = ConvBNActMLX(channels, channels, 1)

        def __call__(self, x):
            return self.pw(self.dw(x))


    class HCDS31MLX(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = ConvBNActMLX(3, 16, 3, stride=2)
            self.s1_down = DownBlockMLX(16, 32)
            self.s1_refine = [ResidualBlockMLX(32)]
            self.s2_down = DownBlockMLX(32, 48)
            self.s2_refine = [ResidualBlockMLX(48) for _ in range(2)]
            self.s3_down = DownBlockMLX(48, 64)
            self.s3_refine = [ResidualBlockMLX(64) for _ in range(3)]
            self.s4_down = DownBlockMLX(64, 96)
            self.s4_refine = [ResidualBlockMLX(96) for _ in range(2)]
            self.lat3 = ConvBNActMLX(64, 32, 1, activate=False)
            self.deep4 = ConvBNActMLX(96, 32, 1, activate=False)
            self.p3_refine = RefineBlockMLX()
            self.lat2 = ConvBNActMLX(48, 32, 1, activate=False)
            self.p2_refine = RefineBlockMLX()
            self.lat1 = ConvBNActMLX(32, 32, 1, activate=False)
            self.p1_refine = RefineBlockMLX()
            self.head = nn.Conv2d(32, 16, 1, bias=True)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        @staticmethod
        def _run(x, blocks):
            for block in blocks:
                x = block(x)
            return x

        def __call__(self, x):
            stem = self.stem(x)
            s1 = self._run(self.s1_down(stem), self.s1_refine)
            s2 = self._run(self.s2_down(s1), self.s2_refine)
            s3 = self._run(self.s3_down(s2), self.s3_refine)
            s4 = self._run(self.s4_down(s3), self.s4_refine)
            p3 = self.p3_refine(self.lat3(s3) + self.upsample(self.deep4(s4)))
            p2 = self.p2_refine(self.lat2(s2) + self.upsample(p3))
            p1 = self.p1_refine(self.lat1(s1) + self.upsample(p2))
            return self.head(p1)

else:
    class HCDS31MLX:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("MLX is unavailable; install it on Apple Silicon with `pip install mlx`")


def mlx_available() -> bool:
    return nn is not None
