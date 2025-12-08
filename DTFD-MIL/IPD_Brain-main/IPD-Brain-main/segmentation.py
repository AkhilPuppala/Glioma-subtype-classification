import cv2
import numpy as np
from skimage.color import rgb2hed

# ---------- PARAMETERS ----------
wsi_path = r'D:\IPD\IPD-Brain-main\IPD-Brain-main\datasets\labelled\IN Brain-0025(b).png'
output_path = "final_segmented_smooth.png"

# Adjustable parameters
min_contour_area = 3000     # ignore very small specks
line_thickness = 50          # contour border thickness
close_kernel = 9            # for connecting nearby tissue
open_kernel = 5             # for removing small noise
gaussian_smooth = 9         # to smooth edges while keeping natural curves

# ---------- STEP 1: READ IMAGE ----------
img = cv2.imread(wsi_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {wsi_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sharp_copy = img.copy()

# ---------- STEP 2: COLOR DECONVOLUTION (H&E) ----------
hed = rgb2hed(img)
hematoxylin = hed[:, :, 0]

# ---------- STEP 3: NORMALIZE + BLUR ----------
hematoxylin_norm = cv2.normalize(hematoxylin, None, 0, 255, cv2.NORM_MINMAX)
hematoxylin_uint8 = np.uint8(hematoxylin_norm)
blur = cv2.GaussianBlur(hematoxylin_uint8, (5, 5), 0)

# ---------- STEP 4: THRESHOLD ----------
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ---------- STEP 5: MORPHOLOGICAL CLEANUP ----------
mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8), iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8), iterations=1)

# ---------- STEP 6: EDGE SMOOTHING ----------
# Gaussian blur to gently smooth the edges without straightening them
mask = cv2.GaussianBlur(mask, (gaussian_smooth, gaussian_smooth), 0)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# ---------- STEP 7: CONTOUR EXTRACTION ----------
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
print(f"Detected {len(contours)} smooth tissue contours")

# ---------- STEP 8: DRAW ON ORIGINAL IMAGE ----------
cv2.drawContours(sharp_copy, contours, -1, (0, 255, 0), line_thickness)

# ---------- STEP 9: SAVE FINAL OUTPUT ----------
cv2.imwrite(output_path, cv2.cvtColor(sharp_copy, cv2.COLOR_RGB2BGR))
print(f"✅ Final smooth segmented slide saved to: {output_path}")
