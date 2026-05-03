from torch.utils.data import Dataset

class MultiViewDataset(Dataset):
  # Argumenty to datasety wczytane osobnymi loaderami
  def __init__(self, original_ds, recons_ds, diff_tensor_ds, diff_image_ds):
    self.original = original_ds
    self.recon = recons_ds
    self.diff_tensor = diff_tensor_ds
    self.diff_image = diff_image_ds
  
  def __len__(self):
    return len(self.original)
  
  @staticmethod
  def extract_base_filename(name: str) -> str:
    name = name.replace("recon_", "").replace("dire_", "")
    return name.rsplit(".", 1)[0]
  
  def __getitem__(self, index):
    x_original, y, original_name = self.original[index]
    x_recon, _, recon_name = self.recon[index]
    x_diff_tensor, _, diff_tensor_name = self.diff_tensor[index]
    x_diff_image, _, diff_image_name = self.diff_image[index]

    base_name = self.extract_base_filename(original_name)
    assert self.extract_base_filename(recon_name) == base_name, (
      f"Recon filename mismatch at idx {index}: {original_name} vs {recon_name}"
    )
    assert self.extract_base_filename(diff_tensor_name) == base_name, (
      f"Diff tensor filename mismatch at idx {index}: {original_name} vs {diff_tensor_name}"
    )
    assert self.extract_base_filename(diff_image_name) == base_name, (
      f"Diff image filename mismatch at idx {index}: {original_name} vs {diff_image_name}"
    )

    return x_original, x_recon, x_diff_tensor, x_diff_image, y, original_name