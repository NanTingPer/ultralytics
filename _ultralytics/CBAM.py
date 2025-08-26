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
    def __init__(self, c1, ksize=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1 , c1 // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 16, c1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=ksize, padding=ksize // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch_att = self.channel_attention(x)
        x = x * ch_att
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        t_cat = torch.cat([avg_pool, max_pool], dim=1)
        sp_input = self.spatial_attention(t_cat)
        sp_att = x * sp_input
        x = x * sp_att
        return x

