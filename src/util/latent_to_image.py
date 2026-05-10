import torch
import torchvision.transforms as T
from pathlib import Path

def latent_to_image(pipe, latent, img_path, prefix: str):
  recon_tensor_ex = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
  recon_tensor_ex = torch.clamp(recon_tensor_ex, -1, 1)
  recon_vis_ex = ((recon_tensor_ex + 1)/2 * 255).clamp(0, 255).to(torch.uint8)
  recon_image_ex = T.ToPILImage()(recon_vis_ex.squeeze(0).cpu())
  recon_image_ex.save(Path("/home/user1/ml-project/data/example") / f"{prefix}_{img_path.stem}.png")