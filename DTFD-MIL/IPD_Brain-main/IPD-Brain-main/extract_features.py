import os
from PIL import Image
import torch
from torchvision import models, transforms
from collections import defaultdict
import re

# -----------------------------
# Configuration
# -----------------------------
input_dir = r"D:\IPD\CLAM-master\datasets\patches_hsv"
output_dir = r"D:\IPD\DTFD-MIL\IPD_Brain-main\IPD-Brain-main\features\pt_files"
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Helper Functions
# -----------------------------

def get_slide_id(filename):
    """Extract slide ID from filename (works for cases with and without parentheses)."""
    base = os.path.splitext(filename)[0]

    # Remove the last two coordinate numbers (_X_Y)
    parts = base.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        slide_id = "_".join(parts[:-2])
    else:
        slide_id = base  # fallback

    return slide_id


def build_model(device):
    """Load pretrained ResNet50 and remove classification layer."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)
    return model


def extract_features(model, imgs, transform):
    """Extract batch features."""
    tensors = [transform(img) for img in imgs]
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        out = model(batch)
        out = out.view(out.size(0), -1)
    return out.cpu()


# -----------------------------
# Group patches by slide prefix
# -----------------------------
files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
if len(files) == 0:
    raise RuntimeError(f"No patch images found in {input_dir}")

slide_groups = defaultdict(list)
for f in files:
    sid = get_slide_id(f)
    slide_groups[sid].append(os.path.join(input_dir, f))

# Sort slides alphabetically for predictable processing order
slide_ids = sorted(slide_groups.keys())

print(f"Found {len(slide_ids)} slides in total.\n")


# -----------------------------
# Setup model and transforms
# -----------------------------
model = build_model(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# Process each slide
# -----------------------------
for slide_id in slide_ids:
    paths = slide_groups[slide_id]
    out_path = os.path.join(output_dir, f"{slide_id}.pt")

    if os.path.exists(out_path):
        print(f"Skipping {slide_id}, already exists.")
        continue

    print(f"\nProcessing {slide_id} ({len(paths)} patches)...")

    imgs = []
    features_list = []

    for i, p in enumerate(paths):
        try:
            imgs.append(Image.open(p).convert('RGB'))
        except Exception as e:
            print(f"⚠️ Failed to open {p}: {e}")
            continue

        # Process batch
        if len(imgs) == batch_size or i == len(paths) - 1:
            feats = extract_features(model, imgs, transform)
            features_list.append(feats)
            imgs = []

    if len(features_list) == 0:
        print(f"⚠️ No valid patches for {slide_id}")
        continue

    feats_all = torch.cat(features_list, dim=0)
    torch.save(feats_all, out_path)
    print(f"✅ Saved: {slide_id} → {feats_all.shape}")
