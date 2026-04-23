from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
import torch
import os

class DIRETensorDataset(Dataset):
  def __init__(self, root_dir, mean=None, std=None):
    self.files = sorted(Path(root_dir).glob("*.pt"))
    self.mean = mean
    self.std = std

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    data = torch.load(self.files[idx], map_location="cpu")

    dire = data["dire"].float()      # (3, H, W)
    label = torch.tensor(data["label"], dtype=torch.float32)
    path = self.files[idx]
    filename = os.path.basename(path)

    # Optional dataset-level normalization
    if self.mean is not None and self.std is not None:
      dire = (dire - self.mean) / (self.std + 1e-6)

    return dire, label, filename


class ResizeWrapper(Dataset):
  def __init__(self, base_dataset, size=(224, 224)):
    self.base_dataset = base_dataset
    self.size = size

  def __len__(self):
    return len(self.base_dataset)

  def __getitem__(self, idx):
    dire, label, filename = self.base_dataset[idx]

    dire = F.interpolate(
        dire.unsqueeze(0),
        size=self.size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    return dire, label, filename