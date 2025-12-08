# import os
# import cv2
# import numpy as np

# # --- Configuration ---
# input_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\labelled"
# output_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\patches_hsv"

# patch_size = 256
# content_threshold = 0.01  
# S_thresh = 40            # minimum saturation (to avoid blank)
# V_thresh = 230           # maximum value (to ignore white areas)

# os.makedirs(output_dir, exist_ok=True)

# for filename in os.listdir(input_dir):
#     if not filename.lower().endswith(('.png', '.jpg', '.tif', '.svs')):
#         continue

#     img_path = os.path.join(input_dir, filename)
#     print(f"Processing {filename}...")

#     # Read image
#     img_bgr = cv2.imread(img_path)
#     if img_bgr is None:
#         print(f"Could not read {filename}")
#         continue

#     # Convert once to RGB and HSV
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

#     # # Save the HSV visualization (optional)
#     # hsv_vis_path = os.path.join(output_dir, filename.replace('.', '_hsv.'))
#     # cv2.imwrite(hsv_vis_path + "png", cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))

#     h, w, _ = img_hsv.shape

#     # Iterate over patches in HSV image
#     for y in range(0, h, patch_size):
#         for x in range(0, w, patch_size):
#             patch_hsv = img_hsv[y:y+patch_size, x:x+patch_size]

#             # Skip incomplete patches at the edges
#             if patch_hsv.shape[0] != patch_size or patch_hsv.shape[1] != patch_size:
#                 continue

#             # Split HSV channels
#             _, s, v = cv2.split(patch_hsv)

#             # Compute tissue mask (S high, V not too high)
#             mask = (s > S_thresh) & (v < V_thresh)
#             content_ratio = np.mean(mask)

#             # If tissue > threshold, map to RGB and save
#             if content_ratio > content_threshold:
#                 patch_rgb = img_rgb[y:y+patch_size, x:x+patch_size]
#                 patch_name = f"{filename.replace('.', '_')}_{x}_{y}.png"
#                 cv2.imwrite(os.path.join(output_dir, patch_name), cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))


import os
import cv2
import numpy as np

# --- Configuration ---
input_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\labelled"
output_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\patches_hsv"

patch_size = 256
non_zero_threshold = 0.065  # Minimum 6.5% non-zero pixels

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.tif', '.svs')):
        continue

    img_path = os.path.join(input_dir, filename)
    print(f"\nProcessing {filename}...")

    # Read image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Could not read {filename}")
        continue

    # Convert to RGB and HSV
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h, w, _ = img_hsv.shape
    h_ch, s_ch, v_ch = cv2.split(img_hsv)

    # --- Step 1: Find valid (non-white) pixels ---
    # Ignore pure white background (v > 250) or very dull pixels (s < 10)
    nonzero_mask = (s_ch > 10) & (v_ch < 250)
    if np.count_nonzero(nonzero_mask) == 0:
        print("⚠️ No valid pixels found, skipping.")
        continue

    nonzero_s = s_ch[nonzero_mask]
    nonzero_v = v_ch[nonzero_mask]

    # --- Step 2: Compute adaptive thresholds using quantiles ---
    # Robust against outliers, adapts per-slide
    s_low = np.percentile(nonzero_s, 20) 
    v_high = np.percentile(nonzero_v, 80) 

    # Add small margins to fine-tune
    S_thresh = np.clip(s_low + 0.1 * s_low, 20, 255)
    V_thresh = np.clip(v_high - 0.05 * v_high, 150, 255)

    print(f"Adaptive thresholds for {filename}: S>{S_thresh:.1f}, V<{V_thresh:.1f}")

    # --- Step 3: Extract and filter patches ---
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch_hsv = img_hsv[y:y+patch_size, x:x+patch_size]

            if patch_hsv.shape[0] != patch_size or patch_hsv.shape[1] != patch_size:
                continue

            _, s_patch, v_patch = cv2.split(patch_hsv)

            # Mask: colorful (S high) and not too bright (V not white)
            mask = (s_patch > S_thresh) & (v_patch < V_thresh)

            non_zero_pixels = np.count_nonzero(mask)
            total_pixels = patch_size * patch_size
            non_zero_ratio = non_zero_pixels / total_pixels

            if non_zero_ratio >= non_zero_threshold:
                patch_rgb = img_rgb[y:y+patch_size, x:x+patch_size]
                patch_name = f"{os.path.splitext(filename)[0]}_{x}_{y}.png"
                cv2.imwrite(os.path.join(output_dir, patch_name),
                            cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))

    print(f"✅ Done: {filename}")
