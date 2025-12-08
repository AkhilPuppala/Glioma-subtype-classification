import torch
import numpy as np
import cv2
import os
import glob
import pandas as pd

# If you use .svs images
try:
    import openslide
    OPENS_SLIDE_ENABLED = True
except ImportError:
    OPENS_SLIDE_ENABLED = False
    print("⚠ OpenSlide not installed. .svs images will not work unless installed.")

# === IMPORT YOUR MODEL CLASSES ===
from Main_DTFD_MIL import (
    Classifier_1fc,
    Attention,
    DimReduction,
    Attention_with_Classifier,
    transform_state_dict,
)

# =======================
#       USER SETTINGS
# =======================

MODEL_PATH = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\Model\best_model1.pth"
COORDS_DIR = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\coords"
FEATURES_DIR = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\features\pt_files"
WSI_DIR = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\labelled"
SAVE_DIR = r"D:\IPD\IPD-Brain-main\IPD-Brain-main\attention_maps"

PATCH_SIZE = 256
DOWNSAMPLE_FACTOR = 4  # Control heatmap resolution (higher = smoother, lower-res)
SLIDE_ID = "IN Brain-0046"

params = {
    "in_chn": 2048,
    "mDim": 384,
    "num_cls": 3,
    "droprate": 0.338125097749074,
    "droprate_2": 0.25,
    "numLayer_Res": 1,
}

# =======================
#     MODEL LOADING
# =======================

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

# =======================
#  LOAD FEATURES + COORDS
# =======================

def load_features_and_coords(slide_id, coords_dir, feat_dir):
    csv_path = os.path.join(coords_dir, f"{slide_id}.csv")
    df = pd.read_csv(csv_path)
    coords = df[['x', 'y']].values.astype(np.int32)

    pt_files = sorted(glob.glob(os.path.join(feat_dir, f"{slide_id}*.pt")))
    assert len(pt_files) > 0, f"❌ No feature .pt files found for {slide_id}"
    feat_list = [torch.load(f) for f in pt_files]
    features = torch.cat(feat_list, dim=0).float()

    assert features.shape[0] == coords.shape[0], \
        f"❌ Mismatch: {features.shape[0]} features vs {coords.shape[0]} coords"

    print(f"📌 Loaded {features.shape[0]} patches for: {slide_id}")
    return features, coords

# =======================
#  GATED ATTENTION COMPUTATION
# =======================

@torch.no_grad()
def get_attention_scores(features, model_dict, device):
    features = features.to(device, non_blocking=True)
    z = model_dict["dim_reduction"](features)
    att_module = model_dict["att_classifier"]

    # ✅ Gated Attention: A = W(vec(tanh(V(z)) * sigmoid(U(z))))
    V = torch.tanh(att_module.attention.attention_V(z))
    U = torch.sigmoid(att_module.attention.attention_U(z))
    A = att_module.attention.attention_weights(V * U)  # [N, 1]
    A = torch.softmax(A.squeeze(1), dim=0)

    return A.cpu().numpy()

# =======================
#  VISUALIZATION
# =======================

def load_slide(slide_id, wsi_dir):
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        img_path = os.path.join(wsi_dir, f"{slide_id}{ext}")
        if os.path.exists(img_path):
            print(f"📌 Loading slide image: {os.path.basename(img_path)}")
            return cv2.imread(img_path)
    raise FileNotFoundError(f"❌ No slide found for {slide_id}")

def visualize_attention(coords, attention_scores, wsi_image, slide_id, patch_size=256, ds_factor=4):
    os.makedirs(SAVE_DIR, exist_ok=True)
    H, W = wsi_image.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)

    # Normalize scales
    A = attention_scores.astype(np.float32)
    A = (A - A.min()) / (A.max() - A.min() + 1e-6)

    # Paint patch contributions (max-merge)
    for (x, y), score in zip(coords, A):
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + patch_size, W), min(y0 + patch_size, H)
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], score)

    # ✅ ✅ Downsample for smoother resolution
    low_res = cv2.resize(
        heatmap, 
        (W // ds_factor, H // ds_factor),
        interpolation=cv2.INTER_AREA
    )
    heatmap = cv2.resize(
        low_res,
        (W, H),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert to heatmap
    heat_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlaid = cv2.addWeighted(wsi_image, 0.6, heatmap_color, 0.4, 0)

    # Save
    overlay_path = os.path.join(SAVE_DIR, f"{slide_id}_heatmap.png")
    cv2.imwrite(overlay_path, overlaid)
    print(f"✅ Saved heatmap overlay: {overlay_path}")

# =======================
#         MAIN
# =======================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n🔧 Using device:", device)

    model_dict = load_model(device)
    features, coords = load_features_and_coords(SLIDE_ID, COORDS_DIR, FEATURES_DIR)
    wsi_image = load_slide(SLIDE_ID, WSI_DIR)

    print("\n🔍 Computing attention scores...")
    attention_scores = get_attention_scores(features, model_dict, device)

    print("\n📊 Attention stats:")
    print("min:", float(attention_scores.min()))
    print("max:", float(attention_scores.max()))
    print("mean:", float(attention_scores.mean()))
    print("std:", float(attention_scores.std()))

    print("\n🎨 Rendering smoothed heatmap...")
    visualize_attention(coords, attention_scores, wsi_image, SLIDE_ID, patch_size=PATCH_SIZE, ds_factor=DOWNSAMPLE_FACTOR)

if __name__ == "__main__":
    main()
