from torchvision import models
import torch.nn as nn
from classifier_CNN import DIRECNNEncoder
import torch
import torch.nn.functional as F

def make_resnet18_encoder():
  model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
  )
  out_dim = model.fc.in_features
  model.fc = nn.Identity()

  return model, out_dim


class MultiViewClassifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.resnet, resnet_dim = make_resnet18_encoder()

    self.diff_tensor_enc = DIRECNNEncoder(in_channels=3)
    self.diff_image_enc = DIRECNNEncoder(in_channels=3)

    # ostateczna ilość kanałów ze wszystkich sieci
    total_dim = resnet_dim * 2 + 128 + 128

    self.classifier = nn.Sequential(
      nn.Linear(total_dim, 512), # Najpierw przechodzimy z total dim na 512 (bo nie wiadomo ile tych cech Resnet miał ostatecznie)
      nn.ReLU(inplace=True),
      nn.Dropout(0.3),
      nn.Linear(512, 1) # przejście z ostatecznej liczby kanałów do wyniku modelu
    )
  
  def forward(self, original, recon, diff_tensor, diff_image):
    z_original = self.resnet(original)
    z_recon = self.resnet(recon)
    z_diff_tensor = self.diff_tensor_enc(diff_tensor)
    z_diff_image = self.diff_image_enc(diff_image)

    # Normalizacja żeby zapobiec dominacji jednej gałęzi
    z_original = F.normalize(z_original, dim=1)
    z_recon = F.normalize(z_recon, dim=1)
    z_diff_tensor = F.normalize(z_diff_tensor, dim=1)
    z_diff_image = F.normalize(z_diff_image, dim=1)

    z = torch.cat(
      [z_original, z_recon, z_diff_tensor, z_diff_image],
      dim=1
    )

    return self.classifier(z)