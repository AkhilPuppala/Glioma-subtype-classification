from __future__ import print_function

import argparse
import os
import numpy as np
import pandas as pd

import torch
from utils.file_utils import save_pkl
from utils.core_utils import train
from utils.utils import seed_torch
from dataset_modules.dataset_generic import Generic_MIL_Dataset

from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------
# LABEL NAMES
# -----------------------------------------------------------
LABEL_MAP = {
    0: "ASTROCYTOMA",
    1: "GLIOBLASTOMA",
    2: "OLIGODENDROGLIOMA"
}


# -----------------------------------------------------------
# SAVE CONFUSION MATRIX
# -----------------------------------------------------------
def save_confusion_matrix(df, path):
    cm = confusion_matrix(df["true_label_num"], df["pred_label_num"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=[LABEL_MAP[i] for i in range(3)],
        yticklabels=[LABEL_MAP[i] for i in range(3)]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# -----------------------------------------------------------
# COLLECT PREDICTIONS FOR TEST SET
# -----------------------------------------------------------
def collect_predictions(model, loader, n_classes):
    model.eval()

    slide_ids = loader.dataset.slide_data["slide_id"]
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)

            logits, prob, Y_hat, _, _ = model(data)

            all_probs.append(prob.cpu().numpy())
            all_labels.append(int(label))
            all_preds.append(int(Y_hat))

    probs = np.vstack(all_probs)

    df = pd.DataFrame({
        "slide_id": slide_ids.values,
        "true_label_num": all_labels,
        "pred_label_num": all_preds,
        "true_label": [LABEL_MAP[x] for x in all_labels],
        "pred_label": [LABEL_MAP[x] for x in all_preds],
    })

    for c in range(n_classes):
        df[f"prob_class_{c}"] = probs[:, c]

    return df


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main(args):

    print(f"[INFO] Seeding everything with seed = {args.seed}")
    seed_torch(args.seed)

    print("[INFO] Loading dataset...")

    dataset = Generic_MIL_Dataset(
        csv_path=r"D:\IPD\CLAM-master\datasets\ipd_clam_ready.csv",
        data_dir=args.data_root_dir,
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

    args.n_classes = 3
    args.reg = 1e-5
    args.subtyping = True
    args.no_inst_cluster = False
    args.weighted_sample = True

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, f"IPD_s{args.seed}")
    os.makedirs(args.results_dir, exist_ok=True)

    pred_dir = os.path.join(args.results_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    folds = np.arange(args.k)
    all_test_auc, all_val_auc = [], []
    all_test_acc, all_val_acc = [], []

    # NEW METRICS
    all_test_recall, all_val_recall = [], []
    all_test_f1, all_val_f1 = [], []
    all_test_precision, all_val_precision = [], []   # <<< ADDED

    print(f"[INFO] Using {args.k}-fold CV")

    for i in folds:

        split_path = os.path.join(args.split_dir, f"splits_{i}.csv")
        print(f"[INFO] Loading split file: {split_path}")

        train_ds, val_ds, test_ds = dataset.return_splits(
            from_id=False, csv_path=split_path
        )

        print(f"[INFO] Training Fold {i}")

        results, test_auc, val_auc, test_acc, val_acc, model = train(
            (train_ds, val_ds, test_ds),
            i,
            args,
            return_model=True
        )

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        torch.save(model.state_dict(), os.path.join(args.results_dir, f"fold_{i}_model.pt"))

        from utils.utils import get_split_loader
        test_loader = get_split_loader(test_ds, testing=False)

        fold_df = collect_predictions(model, test_loader, args.n_classes)
        fold_df.to_csv(os.path.join(pred_dir, f"fold_{i}_predictions.csv"), index=False)

        save_confusion_matrix(fold_df, os.path.join(pred_dir, f"fold_{i}_confusion_matrix.png"))

        # -------------------------------
        # NEW: Recall, F1, Precision
        # -------------------------------
        y_true = fold_df["true_label_num"].values
        y_pred = fold_df["pred_label_num"].values

        test_recall = recall_score(y_true, y_pred, average="macro")
        test_f1 = f1_score(y_true, y_pred, average="macro")
        test_precision = precision_score(y_true, y_pred, average="macro")   # <<< ADDED

        all_test_recall.append(test_recall)
        all_test_f1.append(test_f1)
        all_test_precision.append(test_precision)                           # <<< ADDED

        save_pkl(os.path.join(args.results_dir, f"split_{i}.pkl"), results)

    # -------------------------------------------------------
    # FINAL SUMMARY CSV
    # -------------------------------------------------------
    df = pd.DataFrame({
        "fold": folds,
        "val_auc": all_val_auc,
        "test_auc": all_test_auc,
        "val_acc": all_val_acc,
        "test_acc": all_test_acc,
        "test_precision": all_test_precision,   # <<< ADDED
        "test_recall": all_test_recall,
        "test_f1": all_test_f1
    })

    df.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)
    print("[INFO] Training completed.")


# -----------------------------------------------------------
# ARGUMENT PARSER
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=2048)
parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
parser.add_argument("--data_root_dir", type=str, required=True)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--results_dir", default="./results/mb")
parser.add_argument("--split_dir", type=str, required=True)
parser.add_argument("--model_type", type=str,
                    choices=["clam_sb", "clam_mb"], default="clam_sb")
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--bag_weight", type=float, default=0.7)
parser.add_argument("--inst_loss", type=str, default="ce")
parser.add_argument("--drop_out", type=float, default=0.25)
parser.add_argument("--model_size", type=str, default="small")
parser.add_argument("--B", type=int, default=5)

args = parser.parse_args()

if __name__ == "__main__":
    main(args)
