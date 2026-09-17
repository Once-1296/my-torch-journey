import os
import numpy as np
from PIL import Image

def remove_empty_images(dataset_dir, min_fill_ratio=0.10, dark_threshold=15):
    removed_count = 0
    
    # Walk through all subfolders (roses, tulips, etc.)
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                # Convert image to a NumPy array of RGB values
                # Convert image to a NumPy array of RGB values
                img = Image.open(file_path).convert("RGB")
                img_array = np.array(img)
                
                # Check the maximum value of R, G, or B for each pixel.
                # If the max value is greater than our threshold, it's a "visible" pixel.
                visible_pixels = np.sum(np.max(img_array, axis=-1) > dark_threshold)
                
                total_pixels = img_array.shape[0] * img_array.shape[1]
                
                # If the visible portion is less than 10%, delete it
                if (visible_pixels / total_pixels) < min_fill_ratio:
                    os.remove(file_path)
                    removed_count += 1
                    
    print(f"Mission accomplished! Deleted {removed_count} empty/black images.")

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed_v2"
# Run the function on your processed dataset folder
remove_empty_images(PROCESSED_DIR)

