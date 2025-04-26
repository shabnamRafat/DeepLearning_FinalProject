import glob
import json
import os
from PIL import Image
import numpy as np

# 1) Point this at your masks folder:

mask_folder = os.path.join("..", "Data", "**", "*.png")
mask_paths = glob.glob(os.path.join(mask_folder, "*.png"))

# 2) Extract unique RGB tuples
colors = set()
for mp in mask_paths:
    img = Image.open(mp).convert("RGB")
    arr = np.array(img).reshape(-1, 3)
    for pixel in arr:
        colors.add(tuple(pixel.tolist()))

# 3) Convert each to a HEX string
hex_colors = ['#%02x%02x%02x' % c for c in sorted(colors)]

# 4) Build a mapping: YOU MUST replace "class_name" with the actual label
class_mapping = {hex_code: "class_name" for hex_code in hex_colors}

# 5) Write it out
with open("class_list.json", "w") as f:
    json.dump(class_mapping, f, indent=2)

print("Created class_list.json with the following colors:")
for hc in hex_colors:
    print(" ", hc)
