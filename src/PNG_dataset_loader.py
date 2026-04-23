from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch

class PNGDataset(Dataset):
  def __init__(self, root_dir, label, transform=None):
    self.image_paths = sorted(Path(root_dir).glob("*.png"))
    self.label = label
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, index):
    path = self.image_paths[index]
    image = Image.open(path).convert("RGB")

    if self.transform:
        image = self.transform(image)

    label = torch.tensor(self.label, dtype=torch.float32)
    return image, label, path.name