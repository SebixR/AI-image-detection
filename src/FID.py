import os 
import torch
from pytorch_fid import fid_score

def calculate_fid_between_folders(path_real, path_fake, batch_size=50, device=None):
  if not os.path.exists(path_real) or not os.path.exists(path_fake):
    raise ValueError("Jedna ze ścieżek nie istnieje!")
  
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")

  paths = [path_real, path_fake]

  try:
    fid_value = fid_score.calculate_fid_given_paths(
      paths=paths,
      batch_size=batch_size,
      device=device,
      dims=2048
    )
    return fid_value
  except Exception as e:
    print(f"Error: {e}")
    return None
  
if __name__ == "__main__":
  folder_real = "/home/user1/ml-project/data/original/train/real/for_stable_diffusion_v1-5"
  folder_fake = "/home/user1/ml-project/data/original/train/fake/stable_diffusion_v1-5"

  print("Calculating FID...")
  score = calculate_fid_between_folders(folder_real, folder_fake)

  if score is not None:
    print(f"\nFID score: {score:.4f}")