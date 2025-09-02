import torch
import torch.nn as nn
# source = import ultralytics.nn.modules
# conv.py
# class CBAM

class CBAM(nn.Module):
    """
    在 ultralytics 的 __init__ 中  from ultralytics.nn.modules import CBAM
    在 ultralytics 的 __init__ 中  __all__ => add "CBAM"
    在 修改 ultralytics/cfg/models/11/yolo11.yaml
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.
        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 16, c1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
        # self.channel_attention = ChannelAttention(c1)
        # self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            (torch.Tensor): Attended output tensor.
        """
        # return self.spatial_attention(self.channel_attention(x))
        x = x * self.channel_attention(x)
        pool_avg = torch.mean(x, dim=1, keepdim=True)
        pool_max, _ = torch.max(x, dim=1, keepdim=True)
        pool_cat = torch.cat([pool_avg, pool_max], dim=1)
        return x * self.spatial_attention(pool_cat)