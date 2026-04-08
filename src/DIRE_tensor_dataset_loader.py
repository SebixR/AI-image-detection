from torch.utils.data import Dataset
from pathlib import Path
import torch

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

    # Optional dataset-level normalization
    if self.mean is not None and self.std is not None:
        dire = (dire - self.mean) / (self.std + 1e-6)

    return dire, label