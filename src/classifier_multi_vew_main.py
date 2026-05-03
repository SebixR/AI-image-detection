from test import test_multi
from classifier_multi_view import MultiViewClassifier
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from DIRE_tensor_dataset_loader import DIRETensorDataset
from PNG_dataset_loader import PNGDataset
from torch.utils.data import DataLoader, ConcatDataset
from multi_dataset_loader import MultiViewDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet-18
model = MultiViewClassifier()

# Move model to GPU
model = model.to(device)

# Freeze all layers of ResNet at first
for p in model.resnet.parameters():
  p.requires_grad = False

criterion = nn.BCEWithLogitsLoss() # loss
optimizer = torch.optim.Adam(
  filter(lambda p: p.requires_grad, model.parameters()),
  lr=1e-3,
  weight_decay=1e-4)
scaler = GradScaler()

# Data loader
from torchvision import transforms
transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
])
# Original images
train_dataset_real_original = PNGDataset("path", label=0, transform=transform)
train_dataset_fake_original = PNGDataset("path", label=1, transform=transform)
test_dataset_real_original = PNGDataset("path", label=0, transform=transform)
test_dataset_fake_original = PNGDataset("path", label=1, transform=transform)
train_dataset_original = ConcatDataset([train_dataset_real_original, train_dataset_fake_original])
test_dataset_original = ConcatDataset([test_dataset_real_original, test_dataset_fake_original])

# Reconstructions
train_dataset_real_recon = PNGDataset("path", label=0, transform=transform)
train_dataset_fake_recon = PNGDataset("path", label=1, transform=transform)
test_dataset_real_recon = PNGDataset("path", label=0, transform=transform)
test_dataset_fake_recon = PNGDataset("path", label=1, transform=transform)
train_dataset_recon = ConcatDataset([train_dataset_real_recon, train_dataset_fake_recon])
test_dataset_recon = ConcatDataset([test_dataset_real_recon, test_dataset_fake_recon])

# Diff tensors
train_dataset_real_diff_tensor = DIRETensorDataset("path")
train_dataset_fake_diff_tensor = DIRETensorDataset("path")
test_dataset_real_diff_tensor = DIRETensorDataset("path")
test_dataset_fake_diff_tensor = DIRETensorDataset("path")
train_dataset_diff_tensor = ConcatDataset([train_dataset_real_diff_tensor, train_dataset_fake_diff_tensor])
test_dataset_diff_tensor = ConcatDataset([test_dataset_real_diff_tensor, test_dataset_fake_diff_tensor])

# Diff images
transform_diff = transforms.ToTensor()
train_dataset_real_diff_image = PNGDataset("path", label=0, transform=transform_diff)
train_dataset_fake_diff_image = PNGDataset("path", label=1, transform=transform_diff)
test_dataset_real_diff_image = PNGDataset("path", label=0, transform=transform_diff)
test_dataset_fake_diff_image = PNGDataset("path", label=1, transform=transform_diff)
train_dataset_diff_image = ConcatDataset([train_dataset_real_diff_image, train_dataset_fake_diff_image])
test_dataset_diff_image = ConcatDataset([test_dataset_real_diff_image, test_dataset_fake_diff_image])

# Check lengths
assert len(train_dataset_original) == len(train_dataset_recon)
assert len(train_dataset_original) == len(train_dataset_diff_tensor)
assert len(train_dataset_original) == len(train_dataset_diff_image)

# Final datasets
train_dataset = MultiViewDataset(
  original_ds=train_dataset_original,
  recons_ds=train_dataset_recon,
  diff_tensor_ds=train_dataset_diff_tensor,
  diff_image_ds=train_dataset_diff_image
  )
test_dataset = MultiViewDataset(
  original_ds=test_dataset_original,
  recons_ds=test_dataset_recon,
  diff_tensor_ds=test_dataset_diff_tensor,
  diff_image_ds=test_dataset_diff_image
  )

# Final loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Check sample alignment
for i in range(len(train_dataset)):
  _ = train_dataset[i]
print("Train dataset alignment OK")
for i in range(len(test_dataset)):
  _ = test_dataset[i]
print("Test dataset alignment OK")

# Data for evaluation
history = {
  "loss": [],
  "accuracy": [],
  "precision": [],
  "roc_auc": []
}

# Training loop
num_epochs = 30

scheduler = CosineAnnealingLR(
  optimizer,
  T_max=num_epochs
)

with open("results.txt", "w+") as f:
  for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Unfreeze after a few epochs
    if epoch == 5:
      for p in model.resnet.layer4.parameters():
        p.requires_grad = True
      optimizer = torch.optim.AdamW([
        {"params": model.resnet.layer4.parameters(), "lr": 1e-5},
        {"params": model.diff_tensor_enc.parameters(), "lr": 1e-3},
        {"params": model.diff_image_enc.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
      ])

      scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - epoch)

    for original, recon, diff_tensor, diff_image, labels, filenames in train_loader:
      start_time = time.time()

      original = original.to(device)
      recon = recon.to(device)
      diff_tensor = diff_tensor.to(device)
      diff_image = diff_image.to(device)
      labels = labels.float().to(device).unsqueeze(1)

      optimizer.zero_grad()

      with autocast(device_type="cuda"):
        outputs = model(
          original,
          recon,
          diff_tensor,
          diff_image
        )
        loss = criterion(outputs, labels)
      
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      total_loss += loss.item()

    scheduler.step()
    
    print(f"Epoch {epoch+1}: loss={total_loss / len(train_loader):.4f}")
    f.write(f"Epoch {epoch+1}: loss={total_loss / len(train_loader):.4f}\n")
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.2f}s")
    f.write(f"Time taken: {(end_time - start_time):.2f}s\n")

    # Save the model
    torch.save(model.state_dict(), f"MultiView_model_DIRE_full_{epoch + 1}epochs.pth")

    # Testing
    model.eval()
    test_results = test_multi(f, epoch, model, device)
    history["loss"].append(total_loss)
    history["accuracy"].append(test_results["accuracy"])
    history["precision"].append(test_results["precision"])
    history["roc_auc"].append(test_results["roc_auc"])

# Plots
import matplotlib.pyplot as plt

epochs = range(num_epochs)

plt.figure()
plt.plot(epochs, history["loss"], label="loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.figure()
plt.plot(epochs, history["accuracy"], label="accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.figure()
plt.plot(epochs, history["precision"], label="precision")
plt.title("Precision")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()

plt.show()