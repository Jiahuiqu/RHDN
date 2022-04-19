import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_excitation = nn.Sequential(
            nn.Conv2d(in_planes, int(in_planes // 4), kernel_size=1, stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(int(in_planes // 4), in_planes, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        avg_out = self.channel_excitation(self.avg_pool(x))
        max_out = self.channel_excitation(self.max_pool(x))

        out = torch.add(avg_out, max_out)
        out = torch.sigmoid(out)
        out = torch.mul(x, out)
        out = torch.add(x, out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_mean = torch.mean(x, dim=1, keepdim=True)
        out_max, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([out_mean, out_max], 1)
        out = self.conv(out)
        out = torch.sigmoid(out)
        out = torch.mul(x, out)
        out = torch.add(x, out)
        return out
