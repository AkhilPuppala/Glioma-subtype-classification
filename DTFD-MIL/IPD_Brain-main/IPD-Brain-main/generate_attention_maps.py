import torch
import numpy as np
import cv2
import os
import glob
import pandas as pd
import argparse

# Try to import OpenSlide
try:
    import openslide
    OPENS_SLIDE_ENABLED = True
except ImportError:
    OPENS_SLIDE_ENABLED = False
    print("⚠ OpenSlide not installed. .svs images will not work.")

# === IMPORT MODEL CLASSES ===
from Main_DTFD_MIL import (
    Classifier_1fc,
    Attention,
    DimReduction,
    Attention_with_Classifier,
    transform_state_dict,
)

# ----------------------------------------------------
# ARGUMENT PARSER WITH DEFAULTS MATCHING run_heatmaps
# ----------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--slide_id", type=str, required=True,
                    help="Slide ID such as 'IN Brain-0002'")

parser.add_argument("--coords_dir", type=str,
                    default=r"D:\IPD\CLAM-master\datasets\coords",
                    help="Directory containing slide .csv coordinate files")

parser.add_argument("--feats_dir", type=str,
                    default=r"D:\IPD\CLAM-master\datasets\features\pt_files",
                    help="Directory containing .pt feature files")

parser.add_argument("--wsi_dir", type=str,
                    default=r"D:\IPD\CLAM-master\datasets\labelled",
                    help="Directory containing WSI images")

parser.add_argument("--save_dir", type=str,
                    default=r"D:\IPD\DTFD-MIL\IPD_Brain-main\IPD-Brain-main\attention_maps",
                    help="Directory to save DTFD heatmaps")

args = parser.parse_args()

SLIDE_ID   = args.slide_id
COORDS_DIR = args.coords_dir
FEATURES_DIR = args.feats_dir
WSI_DIR      = args.wsi_dir
SAVE_DIR     = args.save_dir

PATCH_SIZE = 256
DOWNSAMPLE_FACTOR = 4

MODEL_PATH = r"D:\IPD\DTFD-MIL\IPD_Brain-main\IPD-Brain-main\Model--isSaveModel\abc\abc\best_model.pth"

params = {
    "in_chn": 2048,
    "mDim": 384,
    "num_cls": 3,
    "droprate": 0.338125097749074,
    "droprate_2": 0.25,
    "numLayer_Res": 1,
}

# ----------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------
def load_model(device):
    classifier = Classifier_1fc(params["mDim"], params["num_cls"], params["droprate"]).to(device)
    attention = Attention(params["mDim"]).to(device)
    dimReduction = DimReduction(params["in_chn"], params["mDim"], numLayer_Res=params["numLayer_Res"]).to(device)
    attCls = Attention_with_Classifier(L=params["mDim"], num_cls=params["num_cls"], droprate=params["droprate_2"]).to(device)

    print(f"\n📌 Loading trained model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    util_dict = {
        "classifier": classifier,
        "attention": attention,
        "dim_reduction": dimReduction,
        "att_classifier": attCls,
    }

    embed = transform_state_dict(checkpoint, util_dict)

    for key in util_dict.keys():
        util_dict[key].load_state_dict(embed[key], strict=False)
        util_dict[key].eval()

    print("✅ Model successfully loaded!")
    return util_dict


# ----------------------------------------------------
# LOAD FEATURES + COORDS
# ----------------------------------------------------
def load_features_and_coords(slide_id):
    csv_path = os.path.join(COORDS_DIR, f"{slide_id}.csv")
    df = pd.read_csv(csv_path)
    coords = df[['x', 'y']].values.astype(np.int32)

    pt_files = sorted(glob.glob(os.path.join(FEATURES_DIR, f"{slide_id}*.pt")))
    assert pt_files, f"❌ No .pt feature files found for {slide_id}"

    features = torch.cat([torch.load(f) for f in pt_files], dim=0).float()

    assert features.shape[0] == coords.shape[0], \
        f"❌ Feature/coord mismatch: {features.shape[0]} vs {coords.shape[0]}"

    print(f"📌 Loaded {features.shape[0]} patches for: {slide_id}")
    return features, coords


# ----------------------------------------------------
# ATTENTION COMPUTATION
# ----------------------------------------------------
@torch.no_grad()
def get_attention_scores(features, model_dict, device):
    features = features.to(device)
    z = model_dict["dim_reduction"](features)

    att = model_dict["att_classifier"].attention

    V = torch.tanh(att.attention_V(z))
    U = torch.sigmoid(att.attention_U(z))

    A = att.attention_weights(V * U)
    A = torch.softmax(A.squeeze(1), dim=0)

    return A.cpu().numpy()


# ----------------------------------------------------
# LOAD SLIDE IMAGE
# ----------------------------------------------------
def load_slide(slide_id):
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        path = os.path.join(WSI_DIR, f"{slide_id}{ext}")
        if os.path.exists(path):
            return cv2.imread(path)
    raise FileNotFoundError(f"❌ No image found for {slide_id} in {WSI_DIR}")


# ----------------------------------------------------
# VISUALIZATION — Saves: SLIDE_ID_DTFD_heatmap.png
# ----------------------------------------------------
def visualize_attention(coords, attention_scores, wsi_image, slide_id):
    os.makedirs(SAVE_DIR, exist_ok=True)
    H, W = wsi_image.shape[:2]

    heatmap = np.zeros((H, W), dtype=np.float32)

    # Normalize
    A = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-6)

    for (x, y), val in zip(coords, A):
        x1, y1 = min(x + PATCH_SIZE, W), min(y + PATCH_SIZE, H)
        heatmap[y:y1, x:x1] = np.maximum(heatmap[y:y1, x:x1], val)

    low_res = cv2.resize(heatmap, (W // DOWNSAMPLE_FACTOR, H // DOWNSAMPLE_FACTOR), interpolation=cv2.INTER_AREA)
    heatmap = cv2.resize(low_res, (W, H), interpolation=cv2.INTER_LINEAR)

    heat_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    final_overlay = cv2.addWeighted(wsi_image, 0.6, heatmap_color, 0.4, 0)

    # 🔥 Required by batch file
    out_path = os.path.join(SAVE_DIR, f"{slide_id}_DTFD_heatmap.png")
    cv2.imwrite(out_path, final_overlay)

    print(f"✅ Saved heatmap → {out_path}")


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Using:", device)

    model_dict = load_model(device)
    features, coords = load_features_and_coords(SLIDE_ID)
    wsi_image = load_slide(SLIDE_ID)

    print("🔍 Computing attention...")
    att_scores = get_attention_scores(features, model_dict, device)

    visualize_attention(coords, att_scores, wsi_image, SLIDE_ID)


if __name__ == "__main__":
    main()
