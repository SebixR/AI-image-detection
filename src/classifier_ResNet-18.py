import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from DIRE_tensor_dataset_loader import DIRETensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace the final layer (output)
num_features = model.fc.in_features # input feature shape
model.fc = nn.Linear(num_features, 1) # in features and out features (binary output)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss() # loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # optimizer

# Data loader
train_dataset = DIRETensorDataset("path")
val_dataset = DIRETensorDataset("path")

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Training loop
num_epochs = 10
for epoch in range in range(num_epochs):
  model.train()
  total_loss = 0

  for images, labels in train_loader:
    images = images.to(device)
    labels = labels.float().to(device)

    optimizer.zero_grad()

    outputs = model(images).squeeze(1)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  
  print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Inference
model.eval()

with torch.no_grad():
    image = torch.randn(1, 3, 512, 512).to(device)
    logit = model(image)
    prob = torch.sigmoid(logit)
    prediction = (prob > 0.5).int()

print(prob.item(), prediction.item())