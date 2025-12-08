import os
import cv2
import numpy as np
from collections import defaultdict

# --- Configuration ---
patch_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\pg\patches_seg"
output_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\pg\stitches_slides_seg"
patch_size = 256  # must match patch generation size

os.makedirs(output_dir, exist_ok=True)

# --- Group patches by slide name ---
slides = defaultdict(list)

for fname in os.listdir(patch_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    # e.g. slide1_512_256.png
    parts = fname.split('_')
    if len(parts) < 3:
        continue
    slide_name = "_".join(parts[:-2])  # everything before x_y
    slides[slide_name].append(fname)

# --- Stitch each slide ---
for slide_name, patch_files in slides.items():
    print(f"Stitching {slide_name} with {len(patch_files)} patches...")

    coords = []
    for fname in patch_files:
        parts = fname.replace('.png', '').replace('.jpg', '').split('_')
        try:
            x, y = int(parts[-2]), int(parts[-1])
            coords.append((x, y))
        except:
            continue

    if not coords:
        print(f"No valid patches for {slide_name}")
        continue

    # Determine output image size
    max_x = max(x for x, _ in coords) + patch_size
    max_y = max(y for _, y in coords) + patch_size

    stitched = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    # Place each patch
    for fname in patch_files:
        path = os.path.join(patch_dir, fname)
        parts = fname.replace('.png', '').replace('.jpg', '').split('_')
        try:
            x, y = int(parts[-2]), int(parts[-1])
        except:
            continue

        patch = cv2.imread(path)
        if patch is None:
            continue
        stitched[y:y+patch_size, x:x+patch_size] = patch

    # Save stitched image
    out_path = os.path.join(output_dir, f"{slide_name}_stitched.png")
    cv2.imwrite(out_path, stitched)
    print(f"Saved stitched slide: {out_path}")
