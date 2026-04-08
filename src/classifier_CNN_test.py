import torch
import torch.nn.functional as F
from classifier_CNN import DIRECNN
from torch.utils.data import ConcatDataset, DataLoader
from DIRE_tensor_dataset_loader import DIRETensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from PNG_dataset_loader import PNGDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DIRECNN()
model.load_state_dict(
  torch.load("/home/user1/ml-project/CNN_model_tensor_SDv5_20epochs.pth",
             map_location=device)
)
model.to(device)
model.eval() # wyłącza dropout i ustawia BatchNorm w tryb testowy (taki reset chyba po prostu)

# Load test data
def loadPNGs():
  transform = transforms.ToTensor()

  train_fake_dataset = PNGDataset(
    "/home/user1/ml-project/data/original/test/fake/stable_diffusion_v1-5",
    label=1,
    transform=transform
  )

  train_real_dataset = PNGDataset(
    "/home/user1/ml-project/data/original/test/real/for_stable_diffusion_v1-5",
    label=0,
    transform=transform
  )

  return ConcatDataset([train_fake_dataset, train_real_dataset])

def loadDIRETensor():
  test_fake_dataset = DIRETensorDataset("/home/user1/ml-project/data/dire/test/fake/stable_diffusion_v1-5")
  test_real_dataset = DIRETensorDataset("/home/user1/ml-project/data/dire/test/real/for_stable_diffusion_v1-5")

  return ConcatDataset([test_fake_dataset, test_real_dataset])

test_dataset = loadDIRETensor()
# test_dataset = loadPNGs()

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4
)

# Test
all_preds = []
all_labels = []

with torch.no_grad():
  for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    logits = logits.squeeze()

    probs = torch.sigmoid(logits) # [0, 1]
    preds = (probs > 0.5).float() # 0 or 1

    all_preds.append(preds.cpu())
    all_labels.append(y.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

accuracy = (all_preds == all_labels).float().mean()
print(f"Accuracy: {accuracy:.4f}")

print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, digits=4))