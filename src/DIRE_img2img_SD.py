from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
import torch
import PIL.Image
import matplotlib.pyplot as plt
from pathlib import Path

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
  "CompVis/stable-diffusion-v1-4",
  torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config( # a scheduler determines how noise is added and removed
    pipe.scheduler.config
)
pipe.to("cuda")

input_folder = Path("/home/user1/ml-project/data/train/resized")
output_folder = Path("/home/user1/ml-project/data/train/resized/reconstructions")
output_folder.mkdir(exist_ok=True)  # make folder if it doesn't exist

for img_path in input_folder.glob("*.png"):
  print(f"Processing {img_path.name}...")

  original_image = PIL.Image.open(img_path).convert("RGB")

  image: PIL.Image.Image = pipe(
    prompt="Photo of a fish",
    image=original_image,
    strength=0.5
    ).images[0]
  
  output_file = output_folder / f"recon_{img_path.stem}.png"
  plt.imshow(image)
  plt.axis('off')
  plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
  plt.close()


# Different example, without 
# import torch

# # 1. Encode your image to latent
# latent = pipe.vae.encode(img).latent_dist.sample()  # gives z0

# # 2. Sample your own epsilon (noise)
# noise = torch.randn_like(latent)

# # 3. Pick a timestep t
# t = 25  # for example
# alpha_t = pipe.scheduler.alphas_cumprod[t]

# # 4. Compute noisy latent
# z_t = latent * alpha_t.sqrt() + noise * (1 - alpha_t).sqrt()

# # 5. Feed z_t into UNet + scheduler