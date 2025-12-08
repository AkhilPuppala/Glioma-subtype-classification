import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS = os.path.join(ROOT, 'datasets')
SPLITS_IN = os.path.join(DATASETS, 'splits.csv')
PT_DIR = os.path.join(DATASETS, 'features', 'pt_files')
OUT_DIR = os.path.join(DATASETS, 'splits_available')
OUT_SPLITS = os.path.join(OUT_DIR, 'splits.csv')

os.makedirs(OUT_DIR, exist_ok=True)

print('Reading', SPLITS_IN)
# read as plain CSV: keep structure
splits = pd.read_csv(SPLITS_IN, dtype=str)
pt_basenames = set([os.path.splitext(f)[0] for f in os.listdir(PT_DIR)]) if os.path.isdir(PT_DIR) else set()
print('Found', len(pt_basenames), '.pt files')

# For each cell, keep the value only if its basename is in pt_basenames
for col in splits.columns:
    def keep_if_available(val):
        if pd.isna(val):
            return val
        s = str(val).strip()
        if s == '':
            return val
        base = os.path.splitext(s)[0]
        return s if base in pt_basenames else ''
    splits[col] = splits[col].apply(keep_if_available)

splits.to_csv(OUT_SPLITS, index=False)
print('Wrote available splits to', OUT_SPLITS)
