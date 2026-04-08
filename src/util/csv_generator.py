from pathlib import Path
import csv
import shutil

DATA_ROOT = Path("/home/user1/ml-project/data")
IMAGE_DIR = DATA_ROOT / "images"
OUTPUT_CSV = DATA_ROOT / "dataset.csv"

IMAGE_DIR.mkdir(exist_ok=True)

# Wrong
groups = ["original", "recons", "dire"] 
splits = ["train", "val", "test"]
labels = ["real", "fake"]

rows= []
counter = 0

for group in groups:
  for split in splits:
    for label in labels:
      src_dir = DATA_ROOT / group / split / label
      if not src_dir.exists():
        continue

      for img_path in src_dir.iterdir():
        if not img_path.is_file():
          continue
          
        new_image_name = f"{counter:06d}{img_path.suffix}"
        dst_path = IMAGE_DIR / new_image_name

        shutil.copy2(src=img_path, dst=dst_path)

        rows.append({
          "path": f"images/{new_image_name}_{"model-id"}",
          "original_name": img_path.name,
          "recons_name": "",
          "DIRE_tensor_name": "",
          "DIRE_image_name": "",
          "label": label,
          "split": split
        })

        counter += 1

with open(OUTPUT_CSV, "w", newline="") as f:
  writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
  writer.writeheader()
  writer.writerows(rows)

print(f"Saved {counter} images and dataset.csv")