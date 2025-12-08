# optuna_tuner.py
import argparse
import copy
import os
import optuna
from optuna.importance import get_param_importances
from optuna.visualization import plot_param_importances
import sys
import matplotlib.pyplot as plt
import pandas as pd
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Main_DTFD_LR import main, parser


def objective(trial, base_args):

    # ---------------------------
    # 1) Let Optuna sample HPs
    # ---------------------------
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    droprate = trial.suggest_float('droprate', 0.0, 0.3)

    # Scheduler type
    scheduler = trial.suggest_categorical(
        "scheduler", ["none", "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    )

    # Scheduler parameters
    step_size = trial.suggest_int("step_size", 2, 20)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    T_max = trial.suggest_int("T_max", 2, 20)
    patience = trial.suggest_int("patience", 2, 6)

    # ---------------------------
    # 2) Inject into args
    # ---------------------------
    args = copy.deepcopy(base_args)
    args.lr = lr
    args.weight_decay = weight_decay
    args.droprate = droprate

    args.scheduler = scheduler
    args.step_size = step_size
    args.gamma = gamma
    args.T_max = T_max
    args.patience = patience

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

    # ---------------------------
    # 3) Run main()
    # ---------------------------
    try:
        best_auc, test_auc, best_epoch, test_f1, test_acc, val_acc = main(args)
    except Exception as e:
        print(f"⚠️ Trial {trial.number} failed: {e}")
        trial.set_user_attr("error", str(e))
        return 0.0

    trial.set_user_attr("val_acc", float(val_acc))
    trial.set_user_attr("best_auc", float(best_auc))
    trial.set_user_attr("test_f1", float(test_f1))

    print(f"[TRIAL {trial.number}] VAL_ACC = {val_acc:.4f}")

    return float(val_acc)


def run_study(study_name, n_trials, base_args):
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(lambda t: objective(t, base_args), n_trials=n_trials)

    print("\n========== RESULTS ==========")
    best = study.best_trial
    print(f"Best trial: {best.number}, val_acc={best.value:.4f}")
    print("Params:", best.params)

    # ---------------------------
    # SAVE TRIAL RESULTS
    # ---------------------------
    rows = []
    for t in study.trials:
        row = dict(t.params)
        row['value'] = t.value
        row.update(t.user_attrs)
        rows.append(row)

    os.makedirs("results", exist_ok=True)

    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join("results", f"{study_name}_trials.csv")
        df.to_csv(out, index=False)
        print("Trial results saved to:", out)

    # ---------------------------
    # PARAMETER IMPORTANCE
    # ---------------------------
    print("\n========== PARAMETER IMPORTANCE ==========")

    try:
        importances = get_param_importances(study)
        print("Parameter Importance Ranking:")
        for k, v in importances.items():
            print(f"  {k}: {v:.4f}")

        # Save JSON
        json_path = os.path.join("results", f"{study_name}_importance.json")
        with open(json_path, "w") as f:
            json.dump(importances, f, indent=4)
        print("Importance saved to:", json_path)

        # Save CSV
        imp_df = pd.DataFrame(list(importances.items()), columns=["parameter", "importance"])
        imp_csv = os.path.join("results", f"{study_name}_importance.csv")
        imp_df.to_csv(imp_csv, index=False)
        print("Importance saved to:", imp_csv)

        # Plot Importance
        try:
            fig = plot_param_importances(study)
            fig.write_image(os.path.join("results", "importance_plot.png"))
            print("Importance plot saved to: results/importance_plot.png")
        except Exception as e:
            print("Could not save plot (plotly missing). Error:", e)

    except Exception as e:
        print("⚠️ Failed to compute importance:", e)



if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('--trials', default=10, type=int)
    argp.add_argument('--study-name', default='val_acc_opt', type=str)
    known_args, unknown = argp.parse_known_args()
    base_args = parser.parse_args(unknown)
    run_study(known_args.study_name, known_args.trials, base_args)
