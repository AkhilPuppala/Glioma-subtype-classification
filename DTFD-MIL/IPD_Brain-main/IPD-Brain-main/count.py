import pandas as pd

# ---- Load csv files ----
features_df = pd.read_csv(r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\IPD_Brain_available_features.csv")
splits_df   = pd.read_csv(r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\splits.csv")

# Check column names
print("Features columns:", features_df.columns)
print("Splits columns:", splits_df.columns)

# -------------------------------
# Convert wide splits → long format
# -------------------------------
# splits.csv has columns ['train', 'val', 'test']
long_splits = splits_df.melt(
    value_vars=["train", "val", "test"],
    var_name="split",
    value_name="slide_id"
).dropna()

# Ensure slide_id matches type in both
long_splits["slide_id"] = long_splits["slide_id"].astype(str)
features_df["slide_id"] = features_df["slide_id"].astype(str)

# -------------------------------
# Merge splits with features
# -------------------------------
merged = long_splits.merge(features_df, on="slide_id", how="left")

# Your subtype column is "Subtype"
subtype_col = "Subtype"

# -------------------------------
# Count number of subtypes in each split
# -------------------------------
count_table = (
    merged.groupby(["split", subtype_col])
    .size()
    .reset_index(name="count")
    .sort_values(["split", subtype_col])
)

print("\nSubtype distribution across splits:")
print(count_table)

# -------------------------------
# Pivot for nicer display
# -------------------------------
pivot_table = count_table.pivot(
    index=subtype_col,
    columns="split",
    values="count"
).fillna(0)

print("\nPivot format (rows = subtypes, columns = splits):")
print(pivot_table)
