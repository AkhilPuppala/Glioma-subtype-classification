import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# === Paths ===
dataset_csv = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\IPD_Brain.csv"
pt_dir = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\features\pt_files"
out_csv = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\splits.csv"

# === Load dataset ===
df = pd.read_csv(dataset_csv, dtype=str)

# --- Normalize column name ---
if "slide_id" not in df.columns:
    for alt in ["Case Number", "case_number", "Case_Number"]:
        if alt in df.columns:
            df = df.rename(columns={alt: "slide_id"})
            break

# --- Expand multi-line slide_id entries ---
expanded_rows = []
for _, row in df.iterrows():
    slide_ids = str(row["slide_id"]).replace("\r", "").split("\n")
    slide_ids = [s.strip() for s in slide_ids if s.strip()]
    for sid in slide_ids:
        new_row = row.copy()
        new_row["slide_id"] = sid
        expanded_rows.append(new_row)

df = pd.DataFrame(expanded_rows)

# --- Filter slides that actually have .pt feature files ---
pt_bases = [os.path.splitext(f)[0] for f in os.listdir(pt_dir) if f.endswith(".pt")]
pt_bases_set = set(pt_bases)

df = df[df["slide_id"].isin(pt_bases_set)]
slides = df["slide_id"].unique()

print(f"✅ Found {len(slides)} valid slides with matching .pt features")

# --- Create train/val/test splits (80/10/10) ---
train, temp = train_test_split(slides, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# --- Pad for equal DataFrame column length ---
max_len = max(len(train), len(val), len(test))
train = list(train) + [np.nan] * (max_len - len(train))
val = list(val) + [np.nan] * (max_len - len(val))
test = list(test) + [np.nan] * (max_len - len(test))

# --- Save splits ---
split_df = pd.DataFrame({"train": train, "val": val, "test": test})
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
split_df.to_csv(out_csv, index=False)

print(f"\n✅ Created split file successfully: {out_csv}")
print(split_df.head(10))
