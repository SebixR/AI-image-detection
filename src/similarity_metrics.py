import cv2
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
from torchvision import transforms
from PIL import Image

def calculate_ssim(img1_path, img2_path):
  img1 = cv2.imread(img1_path)
  img2 = cv2.imread(img2_path)

  if img1.shape != img2.shape:
    print(f"Incorrect image shapes: {img1.shape} vs {img2.shape}")
    return
  
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  score, diff = ssim(gray1, gray2, full=True)
  return score


def calculate_lpips(img1_path, img2_path):
  loss_fn = lpips.LPIPS(net='alex')

  transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0)
  img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0)

  dist = loss_fn(img1, img2)
  return dist.item()


ssim_score = calculate_ssim(
  "/home/user1/ml-project/data/example/sdv4_turtle.png",
  "/home/user1/ml-project/data/example/25_steps/recon_sdv4_turtle.png"
  )
lpips_dist = calculate_lpips(
  "/home/user1/ml-project/data/example/sdv4_turtle.png",
  "/home/user1/ml-project/data/example/25_steps/recon_sdv4_turtle.png"
  )

print(f"SSIM: {ssim_score}")
print(f"LPIPS: {lpips_dist}")