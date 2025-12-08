import torch
import numpy as np
import cv2
import os
import glob
import pandas as pd

# === Optional: OpenSlide for .svs ===
try:
    import openslide
    OPENS_SLIDE_ENABLED = True
except ImportError:
    OPENS_SLIDE_ENABLED = False
    print("⚠ OpenSlide not installed. .svs images will not work unless installed.")

# === Import model components ===
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
THUMB_SCALE = 0.15  # thumbnail scale for visualization (0.1–0.25 looks good)
SLIDE_ID = "IN Brain-0008(a)"

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
#  GATED ATTENTION
# =======================

@torch.no_grad()
def get_attention_scores(features, model_dict, device):
    features = features.to(device, non_blocking=True)
    z = model_dict["dim_reduction"](features)
    att_module = model_dict["att_classifier"]

    # Gated attention
    V = torch.tanh(att_module.attention.attention_V(z))
    U = torch.sigmoid(att_module.attention.attention_U(z))
    A = att_module.attention.attention_weights(V * U)  # [N,1]
    A = torch.softmax(A.squeeze(1), dim=0)

    return A.cpu().numpy()


# =======================
#  VISUALIZATION (BLOCKY)
# =======================

def load_slide(slide_id, wsi_dir):
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        img_path = os.path.join(wsi_dir, f"{slide_id}{ext}")
        if os.path.exists(img_path):
            print(f"📌 Loading slide image: {os.path.basename(img_path)}")
            return cv2.imread(img_path)
    raise FileNotFoundError(f"❌ No slide found for {slide_id}")


def visualize_attention_blocky(coords, attention_scores, wsi_image, slide_id, patch_size=256, thumb_scale=0.15):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Normalize attention
    A = attention_scores.astype(np.float32)
    A = (A - A.min()) / (A.max() - A.min() + 1e-6)

    # Build patch grid
    grid_w = coords[:, 0].max() // patch_size + 1
    grid_h = coords[:, 1].max() // patch_size + 1
    grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    for (x, y), score in zip(coords, A):
        gx, gy = x // patch_size, y // patch_size
        grid[gy, gx] = score

    # Create color heatmap from the small grid (INTER_NEAREST keeps blockiness)
    heatmap_color = cv2.applyColorMap((grid * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(
        heatmap_color,
        (wsi_image.shape[1], wsi_image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Downsample WSI to thumbnail
    thumb = cv2.resize(wsi_image, (0, 0), fx=thumb_scale, fy=thumb_scale)
    heat_thumb = cv2.resize(heatmap_color, (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay_thumb = cv2.addWeighted(thumb, 0.6, heat_thumb, 0.4, 0)

    # Save blocky overlay and heatmap
    overlay_path = os.path.join(SAVE_DIR, f"{slide_id}_blocky_heatmap.png")
    heat_only_path = os.path.join(SAVE_DIR, f"{slide_id}_blocky_heatmap_only.png")

    cv2.imwrite(overlay_path, overlay_thumb)
    cv2.imwrite(heat_only_path, heat_thumb)
    print(f"✅ Saved blocky heatmap overlay: {overlay_path}")

    # Optional: zoomed patch (like paper inset)
    zoom_y, zoom_x = overlay_thumb.shape[0] // 3, overlay_thumb.shape[1] // 3
    zoom_h, zoom_w = int(overlay_thumb.shape[0] * 0.25), int(overlay_thumb.shape[1] * 0.25)
    zoom_crop = overlay_thumb[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w]
    zoom_crop = cv2.resize(zoom_crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    zoom_path = os.path.join(SAVE_DIR, f"{slide_id}_zoom_blocky.png")
    cv2.imwrite(zoom_path, zoom_crop)
    print(f"✅ Saved zoom-in view: {zoom_path}")


# =======================
#         MAIN
# =======================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Using device:", device)

    model_dict = load_model(device)
    features, coords = load_features_and_coords(SLIDE_ID, COORDS_DIR, FEATURES_DIR)
    wsi_image = load_slide(SLIDE_ID, WSI_DIR)

    print("🔍 Computing attention scores...")
    attention_scores = get_attention_scores(features, model_dict, device)

    print("\n📊 Attention stats:")
    print("min:", float(attention_scores.min()))
    print("max:", float(attention_scores.max()))
    print("mean:", float(attention_scores.mean()))
    print("std:", float(attention_scores.std()))

    print("🎨 Rendering blocky heatmap (paper-style)...")
    visualize_attention_blocky(coords, attention_scores, wsi_image, SLIDE_ID, patch_size=PATCH_SIZE, thumb_scale=THUMB_SCALE)


if __name__ == "__main__":
    main()
