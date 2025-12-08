import pandas as pd
import re

input_csv = r"D:\IPD\CLAM-master\datasets\IPD_Brain_available_features.csv"
output_csv = r"D:\IPD\CLAM-master\datasets\ipd_clam_ready.csv"

df = pd.read_csv(input_csv)

# Extract case_id by removing anything inside parentheses
# Example: IN Brain-0004(a) -> IN Brain-0004
df["case_id"] = df["slide_id"].apply(lambda x: re.sub(r"\(.*?\)", "", x).strip())

# Keep only required columns
df_out = df[["case_id", "slide_id", "Subtype"]]

df_out.to_csv(output_csv, index=False)
print("Saved:", output_csv)
