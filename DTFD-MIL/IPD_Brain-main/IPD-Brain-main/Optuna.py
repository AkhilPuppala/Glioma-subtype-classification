# optuna_tuner.py
import argparse
import copy
import os
import optuna
import sys

# plotting / logging
import matplotlib.pyplot as plt
import pandas as pd

# ensure project root import path (adjust as needed)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Main_DTFD_MIL import main, parser  # main returns 6 values now
from optuna.importance import get_param_importances


def objective(trial, base_args):
    # hyperparameter space
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    droprate = trial.suggest_float('droprate', 0.0, 0.3)

    args = copy.deepcopy(base_args)
    args.lr = lr
    args.weight_decay = weight_decay
    args.droprate = droprate
    args.numGroup = getattr(args, 'numGroup', 3)
    args.numGroup_test = getattr(args, 'numGroup_test', 3)
    args.numLayer_Res = getattr(args, 'numLayer_Res', 1)
    args.EPOCH = 10
    args.device = "cpu"
    args.disable_writer = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not hasattr(args, "num"):
        args.num = 0
    if not hasattr(args, "p_name"):
        args.p_name = f"optuna_trial_{trial.number}"
    if hasattr(args, "splits_dir"):
        args.fold_csv = args.splits_dir

    try:
        best_auc, test_auc, best_epoch, test_f1, test_acc, val_acc = main(args)
    except Exception as e:
        print(f"⚠️ Trial {trial.number} failed: {e}")
        trial.set_user_attr("error", str(e))
        return 0.0

    trial.set_user_attr("val_acc", float(val_acc))
    trial.set_user_attr("best_auc", float(best_auc))
    trial.set_user_attr("test_f1", float(test_f1))

    print(f"[TRIAL {trial.number}] VAL_ACC = {val_acc:.4f}  (best_epoch={best_epoch})")
    return float(val_acc)


def run_study(study_name, n_trials, base_args):
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(lambda t: objective(t, base_args), n_trials=n_trials)

    print("\n========== RESULTS ==========")
    best = study.best_trial
    print(f"Best trial: {best.number}, val_acc={best.value:.4f}")
    print("Params:", best.params)

    # Save trials to CSV
    rows = []
    for t in study.trials:
        row = dict(t.params)
        row['value'] = t.value
        row.update({k: v for k, v in t.user_attrs.items()})
        rows.append(row)

    if rows:
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(rows)
        out_csv = os.path.join(out_dir, f"{study_name}_trials.csv")
        df.to_csv(out_csv, index=False)
        print("Trial results saved to:", out_csv)

    # ------------------------------------------------------------
    # 🔥 PARAMETER IMPORTANCE
    # ------------------------------------------------------------
    print("\n========== PARAMETER IMPORTANCE ==========")
    importances = get_param_importances(study)

    # Print to console
    for key, value in importances.items():
        print(f"{key}: {value:.4f}")

    # Save importance as CSV
    imp_df = pd.DataFrame(list(importances.items()), columns=["parameter", "importance"])
    imp_csv = os.path.join("results", f"{study_name}_importance.csv")
    imp_df.to_csv(imp_csv, index=False)
    print("Parameter importance saved to:", imp_csv)

    # Optional: Save bar plot
    plt.figure(figsize=(6, 4))
    plt.barh(imp_df["parameter"], imp_df["importance"])
    plt.xlabel("Importance")
    plt.title("Optuna Parameter Importance")
    plt.tight_layout()
    plot_path = os.path.join("results", f"{study_name}_importance.png")
    plt.savefig(plot_path)
    print("Parameter importance plot saved to:", plot_path)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('--trials', default=10, type=int)
    argp.add_argument('--study-name', default='val_acc_opt', type=str)
    known_args, unknown = argp.parse_known_args()
    base_args = parser.parse_args(unknown)
    run_study(known_args.study_name, known_args.trials, base_args)
