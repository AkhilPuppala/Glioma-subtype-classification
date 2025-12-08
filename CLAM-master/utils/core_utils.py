import numpy as np
import torch
import torch.nn as nn
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################################
# Weighted Cross-Entropy for imbalance
############################################################
def make_class_weights(split, n_classes):
    # Extract labels and force to integer
    labels = split.slide_data['label'].astype(int).to_numpy()

    # Count samples per class
    counts = np.bincount(labels, minlength=n_classes)

    # Avoid zero division
    counts[counts == 0] = 1

    # Inverse frequency weighting
    weights = 1.0 / counts
    weights = weights / weights.sum()

    return torch.tensor(weights, dtype=torch.float32, device=device)





############################################################
# Accuracy Logger (unchanged)
############################################################
class Accuracy_Logger(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for cls in np.unique(Y):
            mask = Y == cls
            self.data[cls]["count"] += mask.sum()
            self.data[cls]["correct"] += (Y_hat[mask] == Y[mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        return (None if count == 0 else correct / count), correct, count



############################################################
# EarlyStopping (unchanged)
############################################################
class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss



############################################################
# TRAIN FUNCTION — includes weighted CE + safe k
############################################################
def train(datasets, cur, args, return_model=False):
    print(f"\nTraining Fold {cur}!")

    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train','val','test'],
                os.path.join(args.results_dir, f"splits_{cur}.csv"))

    print("Training on", len(train_split))
    print("Validating on", len(val_split))
    print("Testing on", len(test_split))

    ########################################################
    # Weighted Loss to fix Class-2 collapse
    ########################################################
    if args.weighted_sample:
        class_weights = make_class_weights(train_split, args.n_classes)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    ########################################################
    # Safe k_sample: reduce for small bags
    ########################################################
    if args.model_type in ["clam_sb", "clam_mb"]:
        print(f"[INFO] Using safe k_sample: {args.B}")
        safe_k = args.B

        # very important: shrink k_sample automatically for tiny bags
        def patch_k_sample(model):
            model.k_sample = min(safe_k, 4)   # lower max helps stability
            if model.k_sample < 1: model.k_sample = 1

        # instance loss
        if args.inst_loss == "svm":
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2).to(device)
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        model_dict = {
            "dropout": args.drop_out,
            "n_classes": args.n_classes,
            "size_arg": args.model_size,
            "embed_dim": args.embed_dim,
            "instance_loss_fn": instance_loss_fn,
            "k_sample": args.B
        }

        model = CLAM_SB(**model_dict) if args.model_type=="clam_sb" else CLAM_MB(**model_dict)
        _ = model.to(device)
        patch_k_sample(model)

    else:
        model = MIL_fc_mc(dropout=args.drop_out, n_classes=args.n_classes,
                          embed_dim=args.embed_dim).to(device)

    print("Model initialized")
    print_network(model)

    optimizer = get_optim(model, args)

    train_loader = get_split_loader(train_split, training=True,
                                    weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)

    early = EarlyStopping(patience=20, stop_epoch=50)

    for epoch in range(args.max_epochs):

        if args.model_type in ["clam_sb", "clam_mb"]:
            train_loop_clam(epoch, model, train_loader, optimizer,
                            args.n_classes, args.bag_weight,
                            None, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader,
                                 args.n_classes, early, None,
                                 loss_fn, args.results_dir)
        else:
            train_loop(epoch, model, train_loader, optimizer,
                       args.n_classes, None, loss_fn)
            stop = validate(cur, epoch, model, val_loader,
                            args.n_classes, early, None,
                            loss_fn, args.results_dir)

        if stop:
            break

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt"),weights_only=True)
    )

    _, val_err, val_auc, _ = summary(model, val_loader, args.n_classes)
    print(f"Val Error: {val_err:.4f}, AUC: {val_auc:.4f}")

    results, test_err, test_auc, acc_log = summary(model, test_loader, args.n_classes)
    print(f"Test Error: {test_err:.4f}, AUC: {test_auc:.4f}")

    for c in range(args.n_classes):
        acc, correct, count = acc_log.get_summary(c)
        print(f"class {c}: acc {acc}, correct {correct}/{count}")

    if return_model:
        return results, test_auc, val_auc, 1-test_err, 1-val_err, model

    return results, test_auc, val_auc, 1-test_err, 1-val_err



############################################################
# train_loop_clam — auto adjusts k_sample on tiny bags
############################################################
def train_loop_clam(epoch, model, loader, optimizer, n_classes,
                    bag_weight, writer=None, loss_fn=None):

    model.train()
    acc_logger = Accuracy_Logger(n_classes)
    inst_logger = Accuracy_Logger(n_classes)

    train_loss = 0
    inst_loss_total = 0

    print("\n")
    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)

        # Safe k_sample for tiny bags
        if data.size(0) < model.k_sample:
            model.k_sample = max(1, data.size(0))

        logits, Y_prob, Y_hat, _, inst_dict = model(data, label=label,
                                                    instance_eval=True)

        loss_bag = loss_fn(logits, label)
        loss_inst = inst_dict['instance_loss']
        total_loss = bag_weight * loss_bag + (1-bag_weight) * loss_inst

        inst_logger.log_batch(inst_dict["inst_preds"], inst_dict["inst_labels"])
        acc_logger.log(Y_hat, label)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += loss_bag.item()
        inst_loss_total += loss_inst.item()

        if (batch_idx+1) % 20 == 0:
            print(f"batch {batch_idx}, loss {loss_bag.item():.4f}, "
                  f"instance_loss {loss_inst.item():.4f}, "
                  f"k_sample {model.k_sample}, bag {data.size(0)}")

    print(f"Epoch {epoch}, train_loss {train_loss/len(loader):.4f}, "
          f"train_inst_loss {inst_loss_total/len(loader):.4f}")

    for c in range(n_classes):
        acc, correct, count = acc_logger.get_summary(c)
        print(f"class {c}: acc {acc}, correct {correct}/{count}")



############################################################
# validate_clam (unchanged except k_sample safety)
############################################################
def validate_clam(cur, epoch, model, loader, n_classes, early_stopping=None,
                  writer=None, loss_fn=None, results_dir=None):

    model.eval()
    acc_logger = Accuracy_Logger(n_classes)
    inst_logger = Accuracy_Logger(n_classes)

    val_loss = 0
    inst_loss = 0
    inst_count = 0

    probs = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.inference_mode():
        for i,(data,label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)

            # safe k_sample
            if data.size(0) < model.k_sample:
                model.k_sample = max(1, data.size(0))

            logits, Y_prob, Y_hat, _, inst_dict = model(data, label=label,
                                                        instance_eval=True)

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            inst_loss += inst_dict["instance_loss"].item()
            inst_count += 1

            inst_logger.log_batch(inst_dict["inst_preds"], inst_dict["inst_labels"])
            acc_logger.log(Y_hat, label)

            probs[i] = Y_prob.cpu().numpy()
            labels[i] = label.item()

    val_loss /= len(loader)
    inst_loss /= inst_count

    auc = roc_auc_score(labels, probs, multi_class="ovr")

    print(f"\nVal Set: loss={val_loss:.4f}, inst_loss={inst_loss:.4f}, auc={auc:.4f}")
    for c in range(n_classes):
        acc, correct, count = acc_logger.get_summary(c)
        print(f"class {c}: acc {acc}, correct {correct}/{count}")

    if early_stopping:
        early_stopping(epoch, val_loss,
                       model,
                       ckpt_name=os.path.join(results_dir,
                                             f"s_{cur}_checkpoint.pt"))
        if early_stopping.early_stop:
            print("EARLY STOPPING TRIGGERED")
            return True

    return False



############################################################
# summary (unchanged)
############################################################
def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes)
    model.eval()

    probs = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    results = {}

    with torch.inference_mode():
        for i,(data,label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)

            logits, Y_prob, Y_hat,_,_ = model(data)

            probs[i] = Y_prob.cpu().numpy()
            labels[i] = label.item()
            acc_logger.log(Y_hat, label)

            results[slide_ids.iloc[i]] = {
                "prob": probs[i],
                "label": label.item()
            }

    auc = roc_auc_score(labels, probs, multi_class="ovr")
    error = 1 - np.mean(labels == np.argmax(probs, 1))

    return results, error, auc, acc_logger
