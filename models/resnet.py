import torch
import torch.nn as nn

__all__ = ["resnet"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """ResNet Basic Block for CIFAR"""
    expansion = 1

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR-style datasets (depth = 6n+2)"""

    def __init__(self, depth, output_dims=128, num_class=10, multiplier=1):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth should be 6n+2 (e.g., 20, 32, 44, 56, 110)"
        n = (depth - 2) // 6
        block = BasicBlock

        self.in_channels = 16 * multiplier
        self.conv1 = nn.Conv2d(3, 16 * multiplier, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * multiplier)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16 * multiplier, n)
        self.layer2 = self._make_layer(block, 32 * multiplier, n, stride=2)
        self.layer3 = self._make_layer(block, 64 * multiplier, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 64 * multiplier * block.expansion

        # 投影层（embedding）
        self.proj = nn.Linear(self.feat_dim, output_dims)
        # 分类头
        self.classifier = nn.Linear(output_dims, num_class)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_channels, planes, stride, downsample)]
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feat=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        feat = self.proj(x)            # 投影后的嵌入向量
        logits = self.classifier(feat) # 分类结果

        if return_feat:
            return feat, logits
        return logits


def resnet(depth=32, output_dims=128, num_class=10, multiplier=1):
    """Factory function for ResNet (CIFAR version)"""
    return ResNet(depth=depth, output_dims=output_dims, num_class=num_class, multiplier=multiplier)
