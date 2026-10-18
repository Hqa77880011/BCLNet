import numpy as np
from torch import nn
from torch.nn import functional as F


def conv(inputs, outputs, kernel=3):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, padding=kernel//2), nn.GroupNorm(1, outputs), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, channels=3, width=32):
        super().__init__()
        self.layers = nn.Sequential(conv(channels, width), conv(width, width), conv(width, width))

    def forward(self, x):
        return self.layers(x.float())


def dice(p, y):
    return 1 - (2 * (p * y).sum() + 1e-6) / (p.sum() + y.sum() + 1e-6)


def region(p, y):
    return F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), y.float()) + dice(p, y)


def boundary(y, radius=3):
    return (F.max_pool2d(y, 2*radius+1, 1, radius) + F.max_pool2d(-y, 2*radius+1, 1, radius) > 0).float()


def image_demo(seed, n, channels=3, mask=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, channels, 32, 32)).astype("float32")
    y = (x[:, :1] > .5).astype("float32") if mask else x.copy()
    return {"x": x, "y": y, "group": np.arange(n)}
