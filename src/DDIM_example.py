import PIL.Image # pipe returns images in the form of PIL.Image objects
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import matplotlib.pyplot as plt
import time 

pipe = StableDiffusionPipeline.from_pretrained( # from_pretrained - loads a ready-to-use, trained model
    "CompVis/stable-diffusion-v1-4", # stable diffusion released by CompVis
    torch_dtype=torch.float16 # forces the model to use 16-bit floating point precision (I'm guessing the numbers in question are weights)
)
# pipe.scheduler = DDIMScheduler.from_config( # a scheduler determines how noise is added and removed
#     pipe.scheduler.config
# )
pipe.to("cuda") # this would throw an error if cuda was not available

torch.manual_seed(1283012) # sets the global random seed for all pytorch operations
# any random 6 or 7 digit number

for i in range(1, 2):
    start_time = time.time()

    image: PIL.Image.Image = pipe("a cute capybara").images[0] # calls some __call__ method

    torch.cuda.synchronize() # ensures the GPU operations and this Python code are in sync (GPU stuff is asynchronous by default)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Image {i} generated in {elapsed:.2f} seconds")

    # Save image
    plt.imshow(image)
    plt.axis('off')
    plt.savefig("capybara_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)


    torch.manual_seed(torch.initial_seed() + i)

plt.close()