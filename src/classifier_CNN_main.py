from classifier_CNN import DIRECNN, DIRECNN_new
import torch
import random
import numpy as np
import torch.nn as nn
from DIRE_tensor_dataset_loader import DIRETensorDataset
from PNG_dataset_loader import PNGDataset
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import time

# Initialization (for more deterministic results)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = DIRECNN_new()

# Sanity check
x = torch.randn(1, 3, 512, 512)
y = model(x)

print(y.shape)  # should be [1, 1]
if y.shape != torch.Size([1, 1]):
  exit(1)

# Data loader
def loadPNGs():
  transform = transforms.ToTensor() # already normalizes values to [0, 1]

  train_fake_dataset = PNGDataset(
    "/home/user1/ml-project/data/recons/train/fake/full",
    label=1,
    transform=transform
  )

  train_real_dataset = PNGDataset(
    "/home/user1/ml-project/data/recons/train/real/full",
    label=0,
    transform=transform
  )

  return ConcatDataset([train_fake_dataset, train_real_dataset])

def loadDIRETensor():
  train_fake_dataset = DIRETensorDataset("/home/user1/ml-project/data/dire/train/fake/full")
  train_real_dataset = DIRETensorDataset("/home/user1/ml-project/data/dire/train/real/full")

  return ConcatDataset([train_fake_dataset, train_real_dataset])

train_dataset = loadDIRETensor()
# train_dataset = loadPNGs()

train_loader = DataLoader(
  train_dataset,
  batch_size=8,
  shuffle=True,
  num_workers=4,
  pin_memory=True
)
print("Loaded dataset")

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
  model.parameters(),
  lr=1e-3,
  weight_decay=1e-4
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 25
with open("train_results.txt", "w+") as f:
  for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    total_loss = 0.0

    for dire, label, filename in train_loader:
      dire = dire.to(device) # (B, 3, H, W)
      label = label.to(device).unsqueeze(1) # (B, 1)

      optimizer.zero_grad()

      logits = model(dire)
      loss = criterion(logits, label)

      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss={total_loss / len(train_loader):.4f}")
    f.write(f"Epoch {epoch+1}: loss={total_loss / len(train_loader):.4f}\n")
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.2f}s")
    f.write(f"Time taken: {(end_time - start_time):.2f}s\n")

    # Save the model
    torch.save(model.state_dict(), f"CNNnew_model_DIRE_full_{epoch + 1}epochs.pth")

