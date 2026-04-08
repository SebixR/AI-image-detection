import torch.nn as nn

class DIRECNN(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()

    self.net = nn.Sequential(
      # Block 1
      nn.Conv2d(in_channels, out_channels=32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),

      # Block 2
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),

      # Block 3
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),

      # Block 4 (optional but useful)
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),

      # Collapse spacial dimensions
      nn.AdaptiveAvgPool2d(1),
    )
    self.classifier = nn.Linear(128, 1)

  def forward(self, x):
    x = self.net(x)
    return self.classifier(x.flatten(1))