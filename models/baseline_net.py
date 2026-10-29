import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        # 两个 3x3 卷积层
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # 块内残差连接
        return x + self.conv2(self.relu(self.conv1(x)))


class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        # 1. 输入层 (1 个 3x3 卷积)
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

        # 2. 中间层 (3 个 ResBlock，每个包含 2 个卷积，共 6 层)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # 3. 输出层 (1 个 3x3 卷积，输出预测的雨层)
        self.output_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.input_conv(x)
        feat = self.res_blocks(feat)

        # 得到网络预测的雨层
        rain_layer = self.output_conv(feat)

        # 【关键】根据题目思路：原图 - 雨层 = 无雨图
        clean_image = x - rain_layer

        return clean_image
