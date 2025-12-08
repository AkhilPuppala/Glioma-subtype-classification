import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# --- Configuration ---
input_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\labelled"
output_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\patches_adaptive_s_only"
patch_size = 256
min_tissue_ratio = 0.07   # Keep patch if >=7% tissue pixels
sample_fraction = 0.1     # Fraction of pixels sampled for KMeans

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.tif', '.svs')):
        continue

    img_path = os.path.join(input_dir, filename)
    print(f"\nProcessing {filename}...")

    # Read and convert to HSV
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠️ Could not read {filename}")
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, w, _ = img_hsv.shape
    h_ch, s_ch, v_ch = cv2.split(img_hsv)

    # --- Step 1: Remove background using Otsu on V channel ---
    v_blur = cv2.GaussianBlur(v_ch, (5, 5), 0)
    v_thresh_value, _ = cv2.threshold(
        v_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    tissue_mask_v = v_ch < v_thresh_value  # Tissue candidates: not too bright

    if np.count_nonzero(tissue_mask_v) == 0:
        print("⚠️ No non-white tissue candidate pixels found, skipping.")
        continue

    # --- Step 2: Adaptive threshold on S (Saturation) of tissue candidate pixels ---
    s_values = s_ch[tissue_mask_v]
    num_samples = int(len(s_values) * sample_fraction)
    s_sample = shuffle(s_values, random_state=0, n_samples=num_samples)

    # KMeans to cluster saturation values into low and high intensity groups
    kmeans = KMeans(n_clusters=2, random_state=0).fit(s_sample.reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_.flatten()
    low_center, high_center = sorted(cluster_centers)

    # Define threshold slightly above the lower saturation cluster center
    S_thresh = low_center + 0.2 * (high_center - low_center)
    S_thresh = max(10, float(S_thresh))  # Convert to float and ensure minimum

    print(f"Adaptive S threshold: S > {S_thresh:.1f}")

    # --- Step 3: Extract patches with sufficient tissue ---
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch_s = s_ch[y:y + patch_size, x:x + patch_size]
            patch_v = v_ch[y:y + patch_size, x:x + patch_size]

            if patch_s.shape != (patch_size, patch_size):
                continue

            # Mask: remove background via V, tissue via S
            tissue_mask = (patch_v < v_thresh_value) & (patch_s > S_thresh)

            tissue_ratio = np.mean(tissue_mask)
            if tissue_ratio >= min_tissue_ratio:
                patch_rgb = img_rgb[y:y + patch_size, x:x + patch_size]
                patch_name = f"{os.path.splitext(filename)[0]}_{x}_{y}.png"
                cv2.imwrite(os.path.join(output_dir, patch_name),
                            cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))

    print(f"✅ Done: {filename}")
