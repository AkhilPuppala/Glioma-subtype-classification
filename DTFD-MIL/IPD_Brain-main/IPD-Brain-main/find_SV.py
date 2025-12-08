import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # progress bar

# --- Configuration ---
input_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\labelled"
output_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\SV_analysis_results"
os.makedirs(output_dir, exist_ok=True)

patch_size = 256
sample_size = 100  # use first 100 slides
max_dim = 2000     # max resize dimension for large slides
s_percentiles = [5, 8, 10, 12, 15, 17, 20]
v_percentiles = [75, 80, 82, 85, 87, 90, 92, 95]

# --- Select first 100 slides ---
all_slides = sorted([f for f in os.listdir(input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.tif', '.svs'))])
sample_slides = all_slides[:min(sample_size, len(all_slides))]
print(f"🧩 Testing on {len(sample_slides)} slides")

results = []

# --- Process slides with progress bar ---
for filename in tqdm(sample_slides, desc="Analyzing slides", ncols=100):
    img_path = os.path.join(input_dir, filename)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        tqdm.write(f"⚠️ Skipping {filename} (read error)")
        continue

    # --- Memory-safe resize ---
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_size = (int(w * scale), int(h * scale))
        img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]

    # --- Convert to HSV ---
    img_hsv = cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(img_hsv)

    # --- Ignore background (white & dull) ---
    nonzero_mask = (s_ch > 10) & (v_ch < 250)
    if np.count_nonzero(nonzero_mask) == 0:
        tqdm.write(f"⚠️ Skipping {filename} (no tissue detected)")
        continue

    nonzero_s = s_ch[nonzero_mask]
    nonzero_v = v_ch[nonzero_mask]

    # --- Test all S/V percentile pairs ---
    for s_p in s_percentiles:
        for v_p in v_percentiles:
            s_low = np.percentile(nonzero_s, s_p)
            v_high = np.percentile(nonzero_v, v_p)

            S_thresh = np.clip(s_low + 0.1 * s_low, 20, 255)
            V_thresh = np.clip(v_high - 0.05 * v_high, 150, 255)

            mask = (s_ch > S_thresh) & (v_ch < V_thresh)
            total_non_zero = np.count_nonzero(mask)
            total_pixels = mask.size
            avg_non_zero = (total_non_zero / total_pixels) * 100 if total_pixels > 0 else 0

            results.append((filename, s_p, v_p, avg_non_zero))

# --- Aggregate results ---
df = pd.DataFrame(results, columns=["Slide", "S_percentile", "V_percentile", "Avg_nonzero_%"])
agg_df = df.groupby(["S_percentile", "V_percentile"])["Avg_nonzero_%"].mean().reset_index()

# --- Save CSV ---
csv_path = os.path.join(output_dir, "percentile_analysis_table.csv")
agg_df.to_csv(csv_path, index=False)
print(f"\n📄 Saved results table: {csv_path}")

# --- 1️⃣ Combined line plot ---
plt.figure(figsize=(10, 6))

mean_s = agg_df.groupby("S_percentile")["Avg_nonzero_%"].mean().reset_index()
plt.plot(mean_s["S_percentile"], mean_s["Avg_nonzero_%"], marker='o', color='blue', label="S Percentile")

mean_v = agg_df.groupby("V_percentile")["Avg_nonzero_%"].mean().reset_index()
plt.plot(mean_v["V_percentile"], mean_v["Avg_nonzero_%"], marker='s', color='red', label="V Percentile")

plt.title("Effect of HSV Percentiles on Tissue Coverage (Global Pixel-Weighted)")
plt.xlabel("Percentile Value")
plt.ylabel("Average Non-Zero Pixels (%)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plot_path = os.path.join(output_dir, "HSV_percentile_vs_tissue_coverage.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"🖼️ Saved line plot: {plot_path}")

# --- 2️⃣ Generate 2D Heatmap for S vs V ---
pivot_df = agg_df.pivot(index="S_percentile", columns="V_percentile", values="Avg_nonzero_%")

plt.figure(figsize=(8, 6))
plt.imshow(pivot_df, cmap="hot", interpolation="nearest", aspect="auto", origin="lower")
plt.colorbar(label="Average Non-Zero Pixels (%)")
plt.xticks(ticks=np.arange(len(pivot_df.columns)), labels=pivot_df.columns)
plt.yticks(ticks=np.arange(len(pivot_df.index)), labels=pivot_df.index)
plt.xlabel("V Percentile")
plt.ylabel("S Percentile")
plt.title("Heatmap of Average Tissue Coverage Across S/V Percentiles")

heatmap_path = os.path.join(output_dir, "HSV_percentile_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"🔥 Saved heatmap: {heatmap_path}")

# --- 3️⃣ Identify best (S,V) combination ---
best_row = agg_df.loc[agg_df["Avg_nonzero_%"].idxmax()]
best_s = best_row["S_percentile"]
best_v = best_row["V_percentile"]
best_val = best_row["Avg_nonzero_%"]
print(f"\n🏆 Best combination: S={best_s}, V={best_v} → {best_val:.2f}% average tissue coverage")

print("\n✅ Analysis completed successfully (global pixel-weighted + percentile heatmap).")
