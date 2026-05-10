import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)

# Delete the layer responsible for classification
model = nn.Sequential(*list(model.children())[:-1])
model= model.to(device)
model.eval()

preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embeddings(folder_path, label):
  embeddings = []
  labels = []
  filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png'))]

  print(f"Calculating {folder_path}...")
  with torch.no_grad():
    for fname in filenames:
      img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
      img_t = preprocess(img).unsqueeze(0).to(device)
      feat = model(img_t).cpu().numpy().flatten()

      embeddings.append(feat)
      labels.append(label)
  return np.array(embeddings), labels


REAL = 'real'
FAKE = 'fake'

real_embeddings, real_labels = get_embeddings("/home/user1/ml-project/data/original/train/real/for_stable_diffusion_v1-5", REAL)
fake_embeddings, fake_labels = get_embeddings("/home/user1/ml-project/data/original/train/fake/stable_diffusion_v1-5", FAKE)

X = np.vstack((real_embeddings, fake_embeddings))
y = real_labels + fake_labels

print("Calculating t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30) # n-components - liczba wyjściowych wymiarów, perplexity - liczba sąsiadów
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(10, 7))
colors = {REAL: 'blue', FAKE: 'red'}

for label in set(y):
  indices = [i for i, l in enumerate(y) if l == label]
  plt.scatter(X_2d[indices, 0], X_2d[indices, 1], c=colors[label], label=label, alpha=0.6, edgecolors='w')

plt.legend()
plt.title("Wizualizacja t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("tsne_analysis.png", dpi=300, bbox_inches='tight')
print("Saved plot")