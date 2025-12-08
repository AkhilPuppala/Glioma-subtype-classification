import torch
import numpy as np
import cv2
import os
import pandas as pd
import argparse

from models.model_clam import CLAM_SB, CLAM_MB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD TRAINED CLAM MODEL
# ============================================================
def load_clam_model(model_path, model_type="clam_mb",
                    embed_dim=2048, n_classes=3,
                    dropout=0.25, size="small",
                    k_sample=5):

    print(f"\n📌 Loading CLAM model: {model_path}")

    instance_loss_fn = torch.nn.CrossEntropyLoss()

    model_dict = {
        "dropout": dropout,
        "n_classes": n_classes,
        "size_arg": size,
        "embed_dim": embed_dim,
        "instance_loss_fn": instance_loss_fn,
        "k_sample": k_sample
    }

    if model_type == "clam_sb":
        model = CLAM_SB(**model_dict).to(device)
    else:
        model = CLAM_MB(**model_dict).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    print("✅ Model loaded successfully\n")
    return model


# ============================================================
# LOAD FEATURES + COORDS
# ============================================================
def load_features_and_coords(slide_id, coords_dir, feats_dir):
    coord_path = os.path.join(coords_dir, f"{slide_id}.csv")
    coords = pd.read_csv(coord_path)[["x", "y"]].values.astype(np.int32)

    feat_path = os.path.join(feats_dir, f"{slide_id}.pt")
    features = torch.load(feat_path, map_location="cpu").float()

    assert features.shape[0] == coords.shape[0], \
        f"❌ Feature/coord mismatch: {features.shape[0]} vs {coords.shape[0]}"

    print(f"📌 Loaded {features.shape[0]} patches")
    return features, coords


# ============================================================
# GET CLASS-SPECIFIC ATTENTION SCORES (CLAM_MB ONLY)
# ============================================================
@torch.no_grad()
def get_class_attention(model, features):

    # forward pass with attention only
    logits, Y_prob, Y_hat, A_raw, _ = model(
        features.to(device),
        instance_eval=False,
        label=None
    )

    # A_raw shape for CLAM_MB:
    # [n_classes, n_patches]
    attn = A_raw.cpu().numpy()

    # normalize each class head 0–1
    attn_norm = (attn - attn.min(axis=1, keepdims=True)) / \
                (attn.max(axis=1, keepdims=True) - attn.min(axis=1, keepdims=True) + 1e-12)

    return attn_norm, int(Y_hat)


# ============================================================
# BUILD COLOR HEATMAP BASED ON CLASS ATTENTION
# ============================================================
def generate_class_colormap(coords, class_attn,
                            wsi_img, slide_id, out_dir,
                            patch_size=256):

    os.makedirs(out_dir, exist_ok=True)
    H, W = wsi_img.shape[:2]

    # take argmax across classes → class with strongest attention
    patch_classes = np.argmax(class_attn, axis=0)

    COLORS = {
        0: (0, 255, 0),      # green
        1: (0, 255, 255),    # yellow
        2: (255, 0, 0)       # blue
    }

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for (x, y), cls in zip(coords, patch_classes):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + patch_size, y0 + patch_size
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLORS[int(cls)], thickness=-1)

    overlay = cv2.addWeighted(wsi_img, 0.6, canvas, 0.4, 0)

    save_path = os.path.join(out_dir, f"{slide_id}_CLASSMAP.png")
    cv2.imwrite(save_path, overlay)

    print(f"✅ Saved class heatmap: {save_path}")


# ============================================================
# MAIN DRIVER
# ============================================================
def run_class_heatmap(slide_id, model_path, coords_dir,
                      feats_dir, wsi_dir, out_dir,
                      model_type="clam_mb"):

    model = load_clam_model(model_path=model_path, model_type=model_type)

    features, coords = load_features_and_coords(slide_id, coords_dir, feats_dir)
    wsi_img = load_wsi_image(slide_id, wsi_dir)

    class_attn, slide_pred = get_class_attention(model, features)

    print(f"\n📊 Slide prediction: {slide_pred}\n")

    generate_class_colormap(coords, class_attn, wsi_img,
                            slide_id, out_dir)


# ============================================================
# LOAD WSI IMAGE
# ============================================================
def load_wsi_image(slide_id, wsi_dir):
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        path = os.path.join(wsi_dir, f"{slide_id}{ext}")
        if os.path.exists(path):
            return cv2.imread(path)
    raise FileNotFoundError(f"❌ WSI not found for {slide_id}")


# ============================================================
# ARGPARSE
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--slide_id", required=True)
    parser.add_argument("--model_path", default=r"D:\IPD\CLAM-master\results\mb\IPD_s1\fold_2_model.pt")
    parser.add_argument("--coords_dir", default="datasets/coords")
    parser.add_argument("--feats_dir", default="datasets/features/pt_files")
    parser.add_argument("--wsi_dir", default="datasets/labelled")
    parser.add_argument("--out_dir", default=r"D:\IPD\CLAM-master\heatmaps_class")
    parser.add_argument("--model_type", default="clam_mb",
                        choices=["clam_sb", "clam_mb"])

    args = parser.parse_args()

    run_class_heatmap(
        slide_id=args.slide_id,
        model_path=args.model_path,
        coords_dir=args.coords_dir,
        feats_dir=args.feats_dir,
        wsi_dir=args.wsi_dir,
        out_dir=args.out_dir,
        model_type=args.model_type
    )
