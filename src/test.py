import torch
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_score

def test_multi(file, epoch, model, test_loader, device):
  file.write(f"Test for epoch {epoch+1}:\n")

  all_probs = []
  all_preds = []
  all_labels = []
  rows = []

  with torch.no_grad():
    for original, recon, diff_tensor, diff_image, label, filenames in test_loader:
      original = original.to(device)
      recon = recon.to(device)
      diff_tensor = diff_tensor.to(device)
      diff_image = diff_image.to(device)
      label = label.to(device)

      logits = model(original, recon, diff_tensor, diff_image)
      logits = logits.view(-1)

      probs = torch.sigmoid(logits).detach().cpu().view(-1) # [0, 1]
      preds = (probs > 0.5).long() # 0 or 1
      label = label.long()

      for f, p, l in zip(filenames, preds, label):
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
      all_labels.append(label.cpu())

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


import torch
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_score

def test_basic(file, epoch, model, test_loader, device):
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