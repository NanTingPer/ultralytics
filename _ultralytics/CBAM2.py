import torch.nn as nn
import torch

class CBAM2(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 16, c1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel_attention(x)
        pool_avg = torch.mean(x, dim=1, keepdim=True)
        pool_max, _ = torch.max(x, dim=1, keepdim=True)
        pool_cat = torch.cat([pool_avg, pool_max], dim=1)
        return x * self.spatial_attention(pool_cat)