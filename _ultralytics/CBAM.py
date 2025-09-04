import torch.nn as nn
import torch

class CBAM(nn.Module):
    """
    在 ultralytics 的 __init__ 中  from ultralytics.nn.modules import CBAM
    在 ultralytics 的 __init__ 中  __all__ => add "CBAM"
    在 修改 ultralytics/cfg/models/11/yolo11.yaml

    在 nn.tasks中添加CBAM为 CBAM -> nn.modules.conv （训练）
    在 __init__ 中 添加CBAM （导出onnx）
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.
        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.c1 = None
        self.channel_attention = None
        self.spatial_attention = None
        # self.channel_attention = ChannelAttention(c1)
        # self.spatial_attention = SpatialAttention(kernel_size)

    def __init_layer__(self, c1):
        kernel_size = self.kernel_size
        self.channel_attention = nn.Sequential(
            # 计算平均池化特征 提取输入的全局信息
            # [batchsize, RGB, H, W]
            # [batchsize, RGB, 1, 1]
            nn.AdaptiveAvgPool2d(1),
            # 对输入进行降维, 抑制不重要信息 保留重要信息
            nn.Conv2d(c1, c1 // 16, kernel_size=1, bias=False),
            # 插入ReLU非线性变换模块, 让模型能学习特征的依赖关系
            nn.ReLU(inplace=True),
            # 对输入进行升维 "放大" 重要像素的贡献
            # 非重要像素值会被抑制
            nn.Conv2d(c1 // 16, c1, kernel_size=1, bias=False),
            # 生成权重信息
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            (torch.Tensor): Attended output tensor.
        """
        # return self.spatial_attention(self.channel_attention(x))
        if (self.spatial_attention is None) or (self.channel_attention is None):
            self.__init_layer__(x.shape[1])

        x = x * self.channel_attention(x)
        pool_avg = torch.mean(x, dim=1, keepdim=True)
        pool_max, _ = torch.max(x, dim=1, keepdim=True)
        pool_cat = torch.cat([pool_avg, pool_max], dim=1)
        return x * self.spatial_attention(pool_cat)