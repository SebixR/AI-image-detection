from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import torchvision.transforms as T
import PIL.Image
from pathlib import Path

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

pipe = StableDiffusionPipeline.from_pretrained( # from_pretrained - loads a ready-to-use, trained model
    "CompVis/stable-diffusion-v1-4", # stable diffusion released by CompVis
    torch_dtype=torch.float16, # forces the model to use 16-bit floating point precision (I'm guessing the numbers in question are weights)
    safety_checker=None
)
pipe.scheduler = DDIMScheduler.from_config( # a scheduler determines how noise is added and removed
    pipe.scheduler.config
)
pipe.to("cuda")
# pipe.enable_model_cpu_offload()

input_folder = Path("/home/user1/ml-project/data/train/resized/fake")
recons_output_folder = Path("/home/user1/ml-project/data/train/resized/reconstructions")
recons_output_folder.mkdir(exist_ok=True)  # make folder if it doesn't exist
dire_output_folder = Path("/home/user1/ml-project/data/dire")
dire_output_folder.mkdir(exist_ok=True)

# Timesteps
t = 25
pipe.scheduler.set_timesteps(t)

files = list(input_folder.glob("*sdv4*.png"))
total = len(files)
for i, img_path in enumerate(files, start=1):
  print(f"[{i}/{total}] Processing {img_path.name}...")

  original_image = PIL.Image.open(img_path)
  original_tensor = pil_to_tensor(original_image)

  with torch.no_grad(): # tells torch to disable gradient tracking, speeding up the whole process
    # Encode to latent
    latent = pipe.vae.encode(original_tensor).latent_dist.sample() # the encode method returns an object with various parameters of the latent
    latent = latent * pipe.vae.config.scaling_factor # a constant defined within Stable Diffusion - not added if we're manually encoding and diffusing
  
    # Sample noise
    noise = torch.randn_like(latent)
    # Compute noisy latent
    z_t: torch.Tensor = pipe.scheduler.add_noise(
      latent,
      noise,
      pipe.scheduler.timesteps[t-1]
    )

    recon_tensor: torch.Tensor = pipe( # __call__
      prompt="a photo",
      guidance_scale=1.0, # disables prompt guidance (not all models handle 0.0 well, so 1.0 is better)
      latents=z_t,
      num_inference_steps=t,
      # eta=0.0, # defaults to 0 anyways, only for the DDIM scheduler, corresponds to the eta parameter from the DDIM paper
      output_type="pt" # defaults to "pil", can also be "latent" or "np"
      ).images[0] # now returns a tensor with values in range [0, 1]
  
  recon_tensor_scaled = recon_tensor * 2 - 1 # scales to [-1, 1]
  
  # Calculate DIRE
  dire = torch.abs(original_tensor - recon_tensor)

  # Convert DIRE to image and save (similar to what they do in compute_dire.py)
  dire_vis = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8)
  dire_vis = dire_vis.squeeze(0)  # (3, H, W)
  dire_image = T.ToPILImage()(dire_vis.cpu())
  dire_image.save(dire_output_folder / f"dire_{img_path.stem}.png")
  
  # Save the reconstruction
  recon_image: PIL.Image.Image = T.ToPILImage()(recon_tensor)
  output_file = recons_output_folder / f"recon_{img_path.stem}.png"
  recon_image.save(output_file)