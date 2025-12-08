import torch
import numpy as np
import cv2
import os
import pandas as pd
import argparse

# ------------------------------
# Import CLAM model
# ------------------------------
from models.model_clam import CLAM_SB, CLAM_MB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================
# LOAD TRAINED CLAM MODEL
# ===========================================
def load_clam_model(model_path, model_type="clam_sb", embed_dim=1024,
                    n_classes=3, dropout=0.25, size="small", k_sample=5):
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

    print("✅ CLAM model loaded successfully")
    return model


# ===========================================
# LOAD FEATURES + COORDINATES
# ===========================================
def load_features_and_coords(slide_id, coords_dir, feats_dir):
    coord_path = os.path.join(coords_dir, f"{slide_id}.csv")
    df = pd.read_csv(coord_path)
    coords = df[['x', 'y']].values.astype(np.int32)

    feat_path = os.path.join(feats_dir, f"{slide_id}.pt")
    assert os.path.exists(feat_path), f"❌ Missing features: {feat_path}"

    features = torch.load(feat_path, map_location="cpu").float()

    assert features.shape[0] == coords.shape[0], \
        f"❌ {features.shape[0]} features vs {coords.shape[0]} coords mismatch"

    print(f"📌 Loaded {features.shape[0]} patches for slide {slide_id}")
    return features, coords


# ===========================================
# EXTRACT ATTENTION FROM CLAM
# ===========================================
@torch.no_grad()
def get_clam_attention(model, features):

    features = features.to(device)

    logits, Y_prob, Y_hat, A_raw, _ = model(
        features,
        instance_eval=False,
        label=None
    )

    attn = A_raw.squeeze().cpu().numpy()

    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-12)

    return attn, Y_prob.cpu().numpy(), int(Y_hat)


# ===========================================
# LOAD BASE WSI IMAGE
# ===========================================
def load_wsi_image(slide_id, wsi_dir):
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        img_path = os.path.join(wsi_dir, f"{slide_id}{ext}")
        if os.path.exists(img_path):
            print(f"📌 Loaded WSI: {img_path}")
            return cv2.imread(img_path)

    raise FileNotFoundError(f"❌ No WSI image found for slide {slide_id}")


# ===========================================
# GENERATE HEATMAP
# ===========================================
def generate_heatmap(coords, attn, wsi_img,
                     slide_id, save_dir,
                     patch_size=256, downsample=4):

    os.makedirs(save_dir, exist_ok=True)
    H, W = wsi_img.shape[:2]

    A = attn.astype(np.float32)
    A = (A - A.min()) / (A.max() - A.min() + 1e-6)

    heatmap = np.zeros((H, W), dtype=np.float32)

    for (x, y), score in zip(coords, A):
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + patch_size, W), min(y0 + patch_size, H)
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], score)

    low_res = cv2.resize(heatmap, (W // downsample, H // downsample), interpolation=cv2.INTER_AREA)
    heatmap = cv2.resize(low_res, (W, H), interpolation=cv2.INTER_LINEAR)

    heat_uint8 = (heatmap * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(wsi_img, 0.6, heat_color, 0.4, 0)

    save_path = os.path.join(save_dir, f"{slide_id}_CLAM_heatmap.png")
    cv2.imwrite(save_path, overlay)

    print(f"✅ Saved CLAM heatmap: {save_path}")


# ===========================================
# MAIN DRIVER
# ===========================================
def run_clam_heatmap(slide_id,
                     model_path,
                     coords_dir,
                     feats_dir,
                     wsi_dir,
                     out_dir,
                     model_type="clam_sb"):

    model = load_clam_model(
        model_path=model_path,
        model_type=model_type,
        embed_dim=2048,
        n_classes=3,
        dropout=0.25,
        size="small",
        k_sample=5
    )

    features, coords = load_features_and_coords(slide_id, coords_dir, feats_dir)
    wsi_img = load_wsi_image(slide_id, wsi_dir)

    attn, probs, pred = get_clam_attention(model, features)

    print("\n📊 Prediction:")
    print(f"Predicted class = {pred}, Probabilities = {probs}")

    generate_heatmap(coords, attn, wsi_img,
                     slide_id, out_dir,
                     patch_size=256,
                     downsample=4)


# ===========================================
# ARGPARSE ENTRYPOINT
# ===========================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--slide_id", required=True, type=str,
                        help="Slide ID for heatmap generation")

    parser.add_argument("--model_path", type=str, required=False,
                        default=r"D:\IPD\CLAM-master\results\IPD_s1\fold_0_model.pt")

    parser.add_argument("--coords_dir", type=str, required=False,
                        default=r"datasets/coords")

    parser.add_argument("--feats_dir", type=str, required=False,
                        default=r"datasets/features/pt_files")

    parser.add_argument("--wsi_dir", type=str, required=False,
                        default=r"datasets/labelled")

    parser.add_argument("--out_dir", type=str, required=False,
                        default=r"D:\IPD\CLAM-master\heatmaps")

    parser.add_argument("--model_type", type=str, required=False,
                        default="clam_sb", choices=["clam_sb", "clam_mb"])

    args = parser.parse_args()

    run_clam_heatmap(
        slide_id=args.slide_id,
        model_path=args.model_path,
        coords_dir=args.coords_dir,
        feats_dir=args.feats_dir,
        wsi_dir=args.wsi_dir,
        out_dir=args.out_dir,
        model_type=args.model_type
    )
