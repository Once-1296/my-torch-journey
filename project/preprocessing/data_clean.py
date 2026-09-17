from PIL import Image
import os
from rembg import remove
import io

def make_clean_pixel_art(image_path, save_path):
    # 1. Read the image and strip the background
    with open(image_path, 'rb') as i:
        input_data = i.read()
    
    # rembg removes the background and makes it transparent
    subject_only = remove(input_data)
    img = Image.open(io.BytesIO(subject_only)).convert("RGBA")
    
    # 2. Paste the flower onto a solid black background
    # (Black is great because it becomes pure 0s in PyTorch tensors)
    background = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(background, img).convert("RGB")
    
    # 3. Crop to a perfect square (center crop)
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    
    # 4. Shrink to 64x64 using NEAREST (keeps the chunky pixel look)
    img_small = img.resize((64, 64), Image.Resampling.NEAREST)
    
    # 5. Quantize to 16 colors to get that retro vibe
    img_pixel = img_small.quantize(colors=16).convert("RGB")
    
    img_pixel.save(save_path)

# Example usage:
# make_pixel_art("real_rose.jpg", "pixel_rose.png")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed_v2"

DIRS = ["daisy", "roses", "sunflowers", "tulips", "dandelion"]

for dir_name in DIRS:
    raw_dir = RAW_DATA_DIR / dir_name
    processed_dir = PROCESSED_DATA_DIR / dir_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file in raw_dir.glob("*.jpg"):
        save_path = processed_dir / img_file.name.replace(".jpg", ".png")
        make_clean_pixel_art(img_file, save_path)
