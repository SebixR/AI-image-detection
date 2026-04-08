import pyarrow.parquet as pq
from PIL import Image
import io
import os

INPUT = "/home/user1/ml-project/data/test-00000-of-00028.parquet"
OUT_DIR = "/home/user1/ml-project/data/imagenet"
os.makedirs(OUT_DIR, exist_ok=True)

table = pq.read_table(INPUT)
print(table.schema)

# Columns
image_col = table["image"].combine_chunks()
# label_a = table["Label_A"].combine_chunks() # 0 - real, 1 - fake
# label_b = table["Label_B"].combine_chunks() 

bytes_col = image_col.field("bytes")
path_col = image_col.field("path")

for i in range(len(table)):
  img_bytes = bytes_col[i].as_py()
#   la = label_a[i].as_py()
#   lb = label_b[i].as_py()

  img = Image.open(io.BytesIO(img_bytes))

  path = path_col[i].as_py()
  if path:
      base = os.path.splitext(os.path.basename(path))[0]
  else:
      base = f"image_{i}"

# filename = f"{i}_{base}_A{la}_B{lb}.png"
  filename = f"{i}_{base}.png"
  img.save(os.path.join(OUT_DIR, filename))