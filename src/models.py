import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

from src.variables import clip_width, clip_height


class ResidualBlock(nn.Module):
    """残差块，包含两个卷积和批归一化层，以及残差连接"""

    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class SRResNet(nn.Module):
    """SRResNet 模型，作为 SRGAN 的基础生成器
    参数:
      - in_channels: 输入通道数（默认为 3）
      - num_features: 特征层通道数（默认为 64）
      - num_residuals: 残差块的数量（默认为 16）
      - upscale_factor: 放大倍数（默认为 4）
    """

    def __init__(self, in_channels=3, num_features=64, num_residuals=16, upscale_factor=4):
        super(SRResNet, self).__init__()
        # 第一层卷积 + 激活
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # 残差块序列
        residual_blocks = [ResidualBlock(num_features) for _ in range(num_residuals)]
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # 中间卷积层用于整合残差块输出
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # 上采样模块：使用 PixelShuffle 实现
        upsample_layers = []
        # 每次将分辨率扩大2倍
        for _ in range(int(upscale_factor // 2)):
            upsample_layers += [
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        # 最后一层卷积，将特征映射到 RGB 通道
        self.conv3 = nn.Conv2d(num_features, in_channels, kernel_size=9, padding=4)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out = self.residual_blocks(out1)
        out = self.conv2(out)
        out = out + out1  # 残差连接
        out = self.upsample(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input layer: (input_channels x H x W) -> (num_filters x H/2 x W/2)
            nn.Conv2d(input_channels, num_filters, kernel_size=8, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.InstanceNorm2d(num_filters),
            nn.LeakyReLU(0.2),

            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.InstanceNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.InstanceNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.InstanceNorm2d(num_filters * 8),

            nn.Sigmoid(),

        )

    def forward(self, x):
        x = self.model(x)
        return x


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=('conv3_3', 'conv4_3')):
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.layer_name_mapping = {
            '3': "conv1_2",
            '8': "conv2_2",
            '17': "conv3_3",
            '26': "conv4_3",
            '35': "conv5_3"
        }
        self.selected_layers = layers
        self.vgg19 = vgg19
        # 冻结参数
        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in self.vgg19._modules.items():
            x = layer(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.selected_layers:
                    features[layer_name] = x
            # 提前结束
            if len(features) == len(self.selected_layers):
                break
        return features
