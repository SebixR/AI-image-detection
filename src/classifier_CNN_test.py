import torch
from classifier_CNN import DIRECNN
from torch.utils.data import ConcatDataset, DataLoader
from DIRE_tensor_dataset_loader import DIRETensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from PNG_dataset_loader import PNGDataset
from pathlib import Path
from collections import defaultdict

# Load test data
def loadPNGs():
  transform = transforms.ToTensor()

  train_fake_dataset = PNGDataset(
    "/home/user1/ml-project/data/recons/test/fake/full",
    label=1,
    transform=transform
  )

  train_real_dataset = PNGDataset(
    "/home/user1/ml-project/data/recons/test/real/full",
    label=0,
    transform=transform
  )

  return ConcatDataset([train_fake_dataset, train_real_dataset])

def loadDIRETensor():
  test_fake_dataset = DIRETensorDataset("/home/user1/ml-project/data/dire/test/fake/full")
  test_real_dataset = DIRETensorDataset("/home/user1/ml-project/data/dire/test/real/full")

  return ConcatDataset([test_fake_dataset, test_real_dataset])

def test(model_path: str, file) -> None:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = DIRECNN()
  model.load_state_dict(
    torch.load(model_path,
              map_location=device)
  )
  model.to(device)
  model.eval() # wyłącza dropout i ustawia BatchNorm w tryb testowy (taki reset chyba po prostu)

  # test_dataset = loadDIRETensor()
  test_dataset = loadPNGs()

  test_loader = DataLoader(
      test_dataset,
      batch_size=8,
      shuffle=False,
      num_workers=4
  )

  # Test
  all_preds = []
  all_labels = []

  rows = []
  
  with torch.no_grad():
    for x, y, filenames in test_loader:
      x = x.to(device)
      y = y.to(device)

      logits = model(x)
      logits = logits.squeeze()

      probs = torch.sigmoid(logits).detach().cpu().view(-1) # [0, 1]
      preds = (probs > 0.5).float() # 0 or 1
      y = y.cpu().view(-1)

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

      all_preds.append(preds.cpu())
      all_labels.append(y.cpu())

  all_preds = torch.cat(all_preds)
  all_labels = torch.cat(all_labels)

  accuracy = (all_preds == all_labels).float().mean()
  file.write(f"Accuracy: {accuracy:.4f}\n\n")

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


def main():
  folder = Path("/home/user1/ml-project")

  with open("test_results.txt", "w+") as f:
    for file_path in sorted(folder.glob("CNN_model_*"), key=lambda p: int(p.stem.split("_")[-1].replace("epochs", ""))):
      if file_path.is_file():
        f.write(f"File: {file_path}\n")
        test(model_path=file_path, file=f)
        f.write("\n*******************************************************\n")
        print(f"Tested file: {file_path}")

if __name__ == "__main__":
  main()