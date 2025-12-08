import os
import argparse
import numpy as np
import pandas as pd

from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, save_splits

parser = argparse.ArgumentParser(description="Create IPD CLAM splits")
parser.add_argument("--csv_path", type=str,
                    default="datasets/ipd_clam_ready.csv")
parser.add_argument("--out_dir", type=str,
                    default="splits/ipd_subtyping_10fold")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--val_frac", type=float, default=0.1)
parser.add_argument("--test_frac", type=float, default=0.1)
args = parser.parse_args()


print("\n[INFO] Loading dataset...")

dataset = Generic_WSI_Classification_Dataset(
    csv_path=args.csv_path,
    shuffle=False,
    seed=args.seed,
    print_info=True,
    label_dict={
        "ASTROCYTOMA": 0,
        "GLIOBLASTOMA": 1,
        "OLIGODENDROGLIOMA": 2
    },
    label_col="Subtype",
    patient_strat=False,
    ignore=[]
)

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.slide_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

os.makedirs(args.out_dir, exist_ok=True)

print("\n[INFO] Creating splits...")

dataset.create_splits(
    k=args.k,
    val_num=val_num,
    test_num=test_num,
    label_frac=1.0
)

for fold in range(args.k):
    print(f"\n[INFO] Generating fold {fold}")

    dataset.set_splits()

    train_split, val_split, test_split = dataset.return_splits(from_id=True)

    # -------------------------
    # 1. Save the main CSV
    # -------------------------
    splits_path = os.path.join(args.out_dir, f"splits_{fold}.csv")
    save_splits(
        [train_split, val_split, test_split],
        ["train", "val", "test"],
        splits_path
    )

    # -------------------------
    # 2. Save boolean CSV
    # -------------------------
    bool_path = os.path.join(args.out_dir, f"splits_{fold}_bool.csv")
    save_splits(
        [train_split, val_split, test_split],
        ["train", "val", "test"],
        bool_path,
        boolean_style=True
    )

    # -------------------------
    # 3. Save descriptor CSV
    # -------------------------
    desc_df = dataset.test_split_gen(return_descriptor=True)
    desc_path = os.path.join(args.out_dir, f"splits_{fold}_descriptor.csv")
    desc_df.to_csv(desc_path)

print("\n[INFO] All splits created successfully!")
