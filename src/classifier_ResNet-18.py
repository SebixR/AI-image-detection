import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
from DIRE_tensor_dataset_loader import DIRETensorDataset
from DIRE_tensor_dataset_loader import ResizeWrapper
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_score

def test(file, epoch, model, test_loader):
  file.write(f"Test for epoch {epoch+1}:\n")

  all_probs = []
  all_preds = []
  all_labels = []
  rows = []

  with torch.no_grad():
    for x, y, filenames in test_loader:
      x = x.to(device)
      y = y.to(device)

      logits = model(x)
      logits = logits.view(-1)

      probs = torch.sigmoid(logits).detach().cpu().view(-1) # [0, 1]
      preds = (probs > 0.5).long() # 0 or 1
      y = y.long()

      for f, p, l in zip(filenames, preds, y):
        filename_no_ext = f.rsplit(".", 1)[0]
        group = filename_no_ext.rsplit("_", 1)[-1]

        rows.append({
          "filename": f,
          "group": group,
          "label": int(l),
          "prediction": int(p),
          "correct": int(p == l)
        })

      all_probs.append(probs.cpu())
      all_preds.append(preds.cpu())
      all_labels.append(y.cpu())

  all_probs = torch.cat(all_probs)
  all_preds = torch.cat(all_preds)
  all_labels = torch.cat(all_labels)

  accuracy = (all_preds == all_labels).float().mean()
  file.write(f"Accuracy: {accuracy:.4f}\n\n")
  fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
  roc_auc = roc_auc_score(all_labels, all_probs)
  file.write(f"ROC-AUC: {roc_auc:.4f}\n")
  precision = precision_score(all_labels, all_preds)
  file.write(f"Precision: {precision:.4f}\n")

  cm = confusion_matrix(all_labels, all_preds)
  report = classification_report(all_labels, all_preds, digits=4)
  file.write("Confusion Matrix:\n")
  file.write(str(cm) + "\n\n")
  file.write("Classification Report:\n")
  file.write(report + "\n")

  group_stats = defaultdict(lambda: {"total": 0, "correct": 0})

  for r in rows:
    g = r["group"]
    group_stats[g]["total"] += 1
    group_stats[g]["correct"] += r["correct"]
  
  file.write("\nPer-group results:\n")
  for g, s in group_stats.items():
    total = s["total"]
    correct = s["correct"]
    acc = correct / total
    file.write(f"{g}: {correct}/{total} ({acc:.4f})\n")

  file.write("\n*******************************************************\n")

  return {
    "accuracy": accuracy,
    "precision": precision,
    "roc_auc": roc_auc
  }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace the final layer (output)
num_features = model.fc.in_features # input feature shape
model.fc = nn.Linear(in_features=num_features, out_features=1) # in features and out features (binary output)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss() # loss
optimizer = torch.optim.Adam([ # optimizer
    {"params": model.conv1.parameters(), "lr": 1e-5},
    {"params": model.layer1.parameters(), "lr": 1e-5},
    {"params": model.layer2.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 3e-5},
    {"params": model.layer4.parameters(), "lr": 3e-5},
    {"params": model.fc.parameters(),     "lr": 1e-3},
])
scaler = GradScaler()

# Data loader
train_dataset_real = DIRETensorDataset("path")
train_dataset_fake = DIRETensorDataset("path")
test_dataset_real = DIRETensorDataset("path")
test_dataset_fake = DIRETensorDataset("path")

train_dataset = ConcatDataset([train_dataset_fake, train_dataset_real])
test_dataset = ConcatDataset([test_dataset_fake, test_dataset_real])

scaled_train_dataset = ResizeWrapper(train_dataset, size=(224, 224))
scaled_test_dataset = ResizeWrapper(test_dataset, size=(224, 224))

train_loader = DataLoader(
    scaled_train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    scaled_test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Data for evaluation
history = {
  "loss": [],
  "accuracy": [],
  "precision": [],
  "roc_auc": []
}

# Training loop
num_epochs = 30
with open("results.txt", "w+") as f:
  for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
      start_time = time.time()

      images = images.to(device)
      labels = labels.float().to(device)

      optimizer.zero_grad()

      scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
      )

      with autocast(device_type="cuda"):
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
      
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      scheduler.step()

      total_loss += loss.item()
    
    print(f"Epoch {epoch+1}: loss={total_loss / len(train_loader):.4f}")
    f.write(f"Epoch {epoch+1}: loss={total_loss / len(train_loader):.4f}\n")
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.2f}s")
    f.write(f"Time taken: {(end_time - start_time):.2f}s\n")

    # Save the model
    torch.save(model.state_dict(), f"ResNet_model_DIRE_full_{epoch + 1}epochs.pth")

    # Testing
    model.eval()
    test_results = test(f, epoch, model)
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