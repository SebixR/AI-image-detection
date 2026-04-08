from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class PNGDataset(Dataset):
  def __init__(self, root_dir, label, transform=None):
    self.root_dir = root_dir
    self.label = label
    self.transform = transform

    self.image_paths = [
      os.path.join(root_dir, f)
      for f in os.listdir(root_dir)
      if f.endswith(".png")
    ]
  
  def __len__(self):
    return len(self.image_paths)
  
  def __getitem__(self, index):
    img_path = self.image_paths[index]
    image = Image.open(img_path).convert("RGB")

    if self.transform:
      image = self.transform(image)

    label = torch.tensor(self.label, dtype=torch.float16)
    return image, label