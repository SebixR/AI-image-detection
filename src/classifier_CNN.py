import torch.nn as nn
import torch.nn.functional as F

class DIRECNN(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()

    self.net = nn.Sequential(
      # Block 1
      nn.Conv2d(in_channels, out_channels=32, kernel_size=3, padding=1),
      # nn.GroupNorm(num_groups=4, num_channels=32),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),

      # Block 2
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
      # nn.GroupNorm(num_groups=8, num_channels=64),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),

      # Block 3
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
      # nn.GroupNorm(num_groups=8, num_channels=128),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),

      # Block 4 (optional but useful)
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
      # nn.GroupNorm(num_groups=8, num_channels=128),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),

      # Collapse spacial dimensions
      nn.AdaptiveAvgPool2d(1),
    )
    self.classifier = nn.Linear(128, 1)

  def forward(self, x):
    x = self.net(x)
    return self.classifier(x.flatten(1))


class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()

    self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)

    self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x):
    residual = x

    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))

    out = out + residual 
    return F.relu(out)


class DIRECNN_new(nn.Module):
  def __init__(self):
    super().__init__()

    self.stem = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True)
    )

    self.block1 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    self.res1 = ResidualBlock(64)

    self.block2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    self.res2 = ResidualBlock(128)

    self.pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(128, 1)

  def forward(self, x):
    x = self.stem(x)
    x = self.block1(x)
    x = self.res1(x)
    x = self.block2(x)
    x = self.res2(x)
    x = self.pool(x)
    return self.fc(x.flatten(1))