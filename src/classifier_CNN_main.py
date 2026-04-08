from classifier_CNN import DIRECNN
import torch
import torch.nn as nn
from DIRE_tensor_dataset_loader import DIRETensorDataset
from PNG_dataset_loader import PNGDataset
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

# Sanity check
model = DIRECNN()
x = torch.randn(1, 3, 512, 512)
y = model(x)

print(y.shape)  # should be [1, 1]
if y.shape != torch.Size([1, 1]):
  exit(1)

# Data loader
def loadPNGs():
  transform = transforms.ToTensor()

  train_fake_dataset = PNGDataset(
    "/home/user1/ml-project/data/original/train/fake/stable_diffusion_v1-5",
    label=1,
    transform=transform
  )

  train_real_dataset = PNGDataset(
    "/home/user1/ml-project/data/original/train/real/for_stable_diffusion_v1-5",
    label=0,
    transform=transform
  )

  return ConcatDataset([train_fake_dataset, train_real_dataset])

def loadDIRETensor():
  train_fake_dataset = DIRETensorDataset("/home/user1/ml-project/data/dire/train/fake/stable_diffusion_v1-5")
  train_real_dataset = DIRETensorDataset("/home/user1/ml-project/data/dire/train/real/for_stable_diffusion_v1-5")

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

num_epochs = 20
for epoch in range(num_epochs):
  model.train()
  total_loss = 0.0

  for dire, label in train_loader:
    dire = dire.to(device) # (B, 3, H, W)
    label = label.to(device).unsqueeze(1) # (B, 1)

    optimizer.zero_grad()

    logits = model(dire)
    loss = criterion(logits, label)

    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  
  print(f"Epoch {epoch+1}: loss={total_loss / len(train_loader):.4f}")

  # Save the model
  if epoch > 5:
    torch.save(model.state_dict(), f"CNN_model_tensor_SDv5_{epoch + 1}epochs.pth")

