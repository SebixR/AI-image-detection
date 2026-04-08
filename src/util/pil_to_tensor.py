import torch
import torchvision.transforms as T

def pil_to_tensor(pil_img, device="cuda", dtype=torch.float16):
  img = pil_img.convert("RGB")
  
  # Resize to 512x512 (Stable Diffusion default)
  transform = T.Compose([
      T.Resize((512, 512)),
      T.ToTensor(),  # converts to [0,1], shape C,H,W
      T.Normalize([0.5]*3, [0.5]*3)  # scale to [-1,1], which is what the VAE expects
  ])
  
  tensor = transform(img).unsqueeze(0).to(device=device, dtype=dtype)  # shape 1,C,H,W
  return tensor