from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import torchvision.transforms as T
import PIL.Image
from pathlib import Path
import time
from util.pil_to_tensor import pil_to_tensor
from typing import Literal

def calculate_DIRE(input_path: str, recons_path: str, dire_path: str, label: Literal["fake", "real"]) -> None:
  # "CompVis/stable-diffusion-v1-4"
  pipe = StableDiffusionPipeline.from_pretrained( # from_pretrained - loads a ready-to-use, trained model
      "stable-diffusion-v1-5/stable-diffusion-v1-5",
      torch_dtype=torch.float16 # forces the model to use 16-bit floating point precision (I'm guessing the numbers in question are weights)
  )
  pipe.scheduler = DDIMScheduler.from_config( # a scheduler determines how noise is added and removed
      pipe.scheduler.config
  )
  pipe.to("cuda")
  pipe.enable_model_cpu_offload() # offloads some logic to the CPU (might be slower, but GPU isn't running at 100% at all times)

  # Folders
  input_folder = Path(input_path)
  recons_output_folder = Path(recons_path)
  recons_output_folder.mkdir(exist_ok=True)  # make folder if it doesn't exist
  dire_output_folder = Path(dire_path) # for both images and tensors
  dire_output_folder.mkdir(exist_ok=True)

  # Timesteps
  total_steps = 50 # number of reconstruction steps
  steps = 25 # which step to start denoising from
  pipe.scheduler.set_timesteps(total_steps)

  # Prompt (empty in this case)
  text_input = pipe.tokenizer(
    [""], 
    padding="max_length", 
    max_length=pipe.tokenizer.model_max_length, 
    truncation=True, 
    return_tensors="pt"
  )
  with torch.no_grad():
    uncond_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]  # (1, seq_len, 768)

  # Main loop
  files = list(input_folder.glob("*.png"))
  total = len(files)
  for i, img_path in enumerate(files, start=1):
    print(f"[{i}/{total}] Processing {img_path.name}...")
    start_time = time.time()

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
        pipe.scheduler.timesteps[steps] # which noising step to stop at (between 1-1000 for DDIM)
      )

      latents = z_t
      for step_idx, timestep in enumerate(pipe.scheduler.timesteps[steps:]):
        # Predict noise
        noise_pred = pipe.unet(latents, timestep, uncond_embeddings).sample
        # DDIM step
        latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

      # Decode latent
      recon_tensor = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
      recon_tensor = torch.clamp(recon_tensor, -1, 1) # scale from [0, 1] to [-1, 1]
    
    # Calculate DIRE
    dire = torch.abs(original_tensor - recon_tensor)

    # Save the raw DIRE tensor
    LABEL_MAP = {
      "real": 0,
      "fake": 1,
    }
    
    torch.save(
      {
        "dire": dire.squeeze(0).cpu(), # shape = (3, H, W)
        "label": LABEL_MAP[label] # 0 - real, 1 - fake
      },
      dire_output_folder / f"dire_{img_path.stem}.pt"
    )

    # Convert DIRE to image and save (similar to what they do in compute_dire.py)
    dire_vis = (dire / 2 * 255).clamp(0, 255).to(torch.uint8)
    dire_vis = dire_vis.squeeze(0)  # (3, H, W)
    dire_image = T.ToPILImage()(dire_vis.cpu())
    dire_image.save(dire_output_folder / f"dire_{img_path.stem}.png")
    
    # Save the reconstruction
    recon_vis = ((recon_tensor + 1)/2 * 255).clamp(0, 255).to(torch.uint8)
    recon_image = T.ToPILImage()(recon_vis.squeeze(0).cpu())
    recon_image.save(recons_output_folder / f"recon_{img_path.stem}.png")

    torch.cuda.synchronize() # ensures the GPU operations and this Python code are in sync (GPU stuff is asynchronous by default)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Time taken: {elapsed:.2f} seconds")

def main():
  calculate_DIRE(
    input_path="/home/user1/ml-project/data/original/test/real/for_stable_diffusion_v1-5",
    recons_path="/home/user1/ml-project/data/recons/test/real/for_stable_diffusion_v1-5",
    dire_path="/home/user1/ml-project/data/dire/test/real/for_stable_diffusion_v1-5",
    label="real"
  )

if __name__ == "__main__":
  main()