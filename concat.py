import sys
from PIL import Image, ImageDraw, ImageFont

clam_img_path = sys.argv[1]
dtfd_img_path = sys.argv[2]
out_path = sys.argv[3]

# Load images
clam = Image.open(clam_img_path)
dtfd = Image.open(dtfd_img_path)

# Match height
h = min(clam.height, dtfd.height)
clam = clam.resize((int(clam.width * h / clam.height), h))
dtfd = dtfd.resize((int(dtfd.width * h / dtfd.height), h))

# Extremely large font (≈5x the previous size)
try:
    font_bottom = ImageFont.truetype("arial.ttf", 200)   # huge text
except:
    font_bottom = ImageFont.load_default()

# Bottom bar height (must be big to hold the large text)
bottom_bar = 280
border_width = 10

# Canvas
w = clam.width + dtfd.width + border_width
canvas = Image.new("RGB", (w, h + bottom_bar), "white")

draw = ImageDraw.Draw(canvas)

# Paste images
canvas.paste(clam, (0, 0))
canvas.paste(dtfd, (clam.width + border_width, 0))

# Vertical border
draw.rectangle(
    [(clam.width, 0), (clam.width + border_width, h)],
    fill="black"
)

# ----- Large Bottom Labels -----
clam_text = "CLAM MODEL"
dtfd_text = "DTFD MODEL"

# CLAM bottom text (centered under left image)
clam_text_w = draw.textlength(clam_text, font=font_bottom)
clam_x = clam.width//2 - clam_text_w//2
draw.text((clam_x, h + 40), clam_text, fill="black", font=font_bottom)

# DTFD bottom text (centered under right image)
dtfd_text_w = draw.textlength(dtfd_text, font=font_bottom)
dtfd_x = clam.width + border_width + (dtfd.width//2 - dtfd_text_w//2)
draw.text((dtfd_x, h + 40), dtfd_text, fill="black", font=font_bottom)

# Save
canvas.save(out_path)
print("Saved:", out_path)
