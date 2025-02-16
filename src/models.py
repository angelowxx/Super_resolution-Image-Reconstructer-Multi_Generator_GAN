import torch
import torch.nn as nn
import torchvision.models as models


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
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # Hidden layers: progressively downsample
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: (num_filters * 8 x H/16 x W/16) -> (1 x H/32 x W/32)
            nn.Conv2d(num_filters * 8, num_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # 自适应池化，固定输出尺寸，例如 (4, 4)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(num_filters * 4 * 4, num_filters),
            nn.BatchNorm1d(num_filters),
            nn.LeakyReLU(0.2),

            nn.Linear(num_filters, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = self.global_pool(x)  # 变成固定大小
        x = self.classifier(x)  # 通过全连接层
        return x


class PerceptualLoss(nn.Module):
    def __init__(self, model_type='vgg16', layer_index=5, device='cpu'):
        super(PerceptualLoss, self).__init__()
        self.device = device
        if model_type == 'vgg19':
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features  # 预训练模型
        elif model_type == 'vgg16':
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features  # 预训练模型
        else:
            raise ValueError("Only 'vgg19' and 'vgg16' are supported.")

        self.feature_extractor = nn.Sequential(*list(vgg[:layer_index])).to(self.device)  # 提取前 layer_index 层
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # 冻结权重，不参与训练

    def forward(self, sr_image, hr_image):
        sr_features = self.feature_extractor(sr_image)
        hr_features = self.feature_extractor(hr_image)
        # 计算混合损失
        loss = nn.functional.mse_loss(sr_features, hr_features) + nn.functional.l1_loss(sr_image, hr_image)
        return loss
