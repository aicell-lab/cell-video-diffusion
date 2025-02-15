import os, glob, random
import tifffile
import numpy as np
from PIL import Image

RAW_DIR = "data/raw/mitotic_cell_atlas_v1.0.1_fulldata/Data_tifs"
OUT_DIR = "data/processed/mca_frame0_256"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(OUT_DIR, "train")
TEST_DIR = os.path.join(OUT_DIR, "test")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Get all cell directories containing TIF files
cell_dirs = sorted(glob.glob(os.path.join(RAW_DIR, "*", "cell*", "conctif")))

random.seed(42)
random.shuffle(cell_dirs)
split_idx = int(0.8 * len(cell_dirs))
train_cells = cell_dirs[:split_idx]
test_cells = cell_dirs[split_idx:]

def process_first_tif(path, out_dir, cell_name):
    """
    Process a single TIF file into a standardized PNG format:
    1. Read the TIF file
    2. If 3D (z-stack), create Maximum Intensity Projection
    3. Normalize pixel values to [0,1] range
    4. Convert to 8-bit (0-255) format
    5. Resize to 256x256 pixels
    6. Save as PNG
    """
    arr = tifffile.imread(path)
    if arr.ndim == 3:  # (z, h, w) => MIP
        arr = np.max(arr, axis=0)
    # Normalize to [0,1] range
    arr = arr - arr.min()
    arr /= (arr.max() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    assert img.size == (256, 256), f"Expected image size (256, 256) but got {img.size}"
    out_name = f"{cell_name}.png"
    img.save(os.path.join(out_dir, out_name))

for d in train_cells:
    tifs = sorted(glob.glob(os.path.join(d, "TR1_*.tif")))
    if tifs:
        cell_name = os.path.basename(os.path.dirname(d))
        process_first_tif(tifs[0], TRAIN_DIR, cell_name)  # Process only first timepoint

for d in test_cells:
    tifs = sorted(glob.glob(os.path.join(d, "TR1_*.tif")))
    if tifs:
        cell_name = os.path.basename(os.path.dirname(d))
        process_first_tif(tifs[0], TEST_DIR, cell_name)  # Process only first timepoint
