import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS = os.path.join(ROOT, 'datasets')
INPUT_CSV = os.path.join(DATASETS, 'IPD_Brain.csv')
PT_DIR = os.path.join(DATASETS, 'features', 'pt_files')
OUT_CSV = os.path.join(DATASETS, 'IPD_Brain_available_features.csv')

print('Reading:', INPUT_CSV)
df = pd.read_csv(INPUT_CSV, dtype=str)

if 'slide_id' not in df.columns:
    # Attempt to rename common alternatives
    for alt in ['Case Number', 'case_number', 'Case_Number']:
        if alt in df.columns:
            df = df.rename(columns={alt: 'slide_id'})
            break

# Gather all available .pt files
pt_basenames = set([os.path.splitext(f)[0] for f in os.listdir(PT_DIR)]) if os.path.isdir(PT_DIR) else set()
print(f'Found {len(pt_basenames)} .pt files in {PT_DIR}')

rows = []
missing_slides = []

for _, r in df.iterrows():
    sid = str(r.get('slide_id', ''))
    parts = [p.strip() for p in sid.replace('\r', '').split('\n') if p.strip()]
    if not parts:
        continue
    for part in parts:
        base = os.path.splitext(part)[0]
        if base in pt_basenames:
            new_r = r.copy()
            new_r['slide_id'] = base
            rows.append(new_r)
        else:
            missing_slides.append(base)

if len(rows) == 0:
    print('No matching slides with .pt found. Exiting without writing file.')
else:
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f'✅ Wrote {OUT_CSV} with {len(out_df)} matching rows')

# Report missing information
unique_missing = sorted(set(missing_slides))
print(f'\nSummary:')
print(f'  Total slides in CSV: {len(df)}')
print(f'  Matching slides found: {len(out_df)}')
print(f'  Missing slides: {len(unique_missing)}')

if unique_missing:
    print('\nList of missing slides:')
    for m in unique_missing:
        print(' ', m)
