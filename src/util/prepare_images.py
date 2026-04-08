from PIL import Image
from pathlib import Path

INPUT_DIR = Path("/home/user1/ml-project/data/original/test/real/full_raw")
OUTPUT_DIR = Path("/home/user1/ml-project/data/original/test/real/full")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SIZE = 512

total = len([p for p in INPUT_DIR.iterdir() if p.is_file()])
processed = 0

for img_path in INPUT_DIR.iterdir():
  if not img_path.is_file():
    continue

  try:
    with Image.open(img_path) as img:
      img = img.convert("RGB")

      width, height = img.size

      # Scale
      if width < height:
        new_width = TARGET_SIZE
        new_height = int(height * TARGET_SIZE / width)
      else:
        new_height = TARGET_SIZE
        new_width = int(width * TARGET_SIZE / height)
      
      img = img.resize((new_width, new_height), Image.LANCZOS)

      # Crop
      left = (new_width - TARGET_SIZE) // 2 # discards reminder
      top = (new_height - TARGET_SIZE) // 2
      right = left + TARGET_SIZE
      bottom = top + TARGET_SIZE

      img = img.crop((left, top, right, bottom))

      # Save
      output_path = OUTPUT_DIR / f"{img_path.stem}.png"
      img.save(output_path, format="PNG")

      processed += 1
      print(f"[{processed}/{total}] Processed: {img_path.name}")
      
  except Exception as e:
    print(f"Error for {img_path.name}: {e}")

