import torch

data = torch.load(
    "/home/user1/ml-project/data/dire/train/real/for_stable_diffusion_v1-5/dire__4814_w600_h600_crop.pt",
    map_location="cpu"
)

print(type(data))
print(data)