import os
import glob
import random
import tifffile
import numpy as np
from PIL import Image

RAW_DIR = "data/raw/mitotic_cell_atlas_v1.0.1_fulldata/Data_tifs"
OUT_DIR = "data/processed/mca_frames_256"  # Changed the folder name for clarity
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(OUT_DIR, "train")
TEST_DIR = os.path.join(OUT_DIR, "test")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Specify which frames to grab for each cell
FRAMES_TO_GRAB = [0, 9, 19, 29, 39]

# Get all cell directories containing TIF files
cell_dirs = sorted(glob.glob(os.path.join(RAW_DIR, "*", "cell*", "conctif")))

random.seed(42)
random.shuffle(cell_dirs)
split_idx = int(0.8 * len(cell_dirs))
train_cells = cell_dirs[:split_idx]
test_cells = cell_dirs[split_idx:]

def process_tif(path, out_dir, file_prefix):
    """
    Process a single TIF file into a standardized PNG format:
    1. Read the TIF file
    2. If 3D (z-stack), create Maximum Intensity Projection
    3. Normalize pixel values to [0,1] range
    4. Convert to 8-bit (0-255) format
    5. Assert it’s 256x256
    6. Save as PNG
    """
    arr = tifffile.imread(path)
    if arr.ndim == 3:  # (z, h, w) => MIP
        arr = np.max(arr, axis=0)

    # Normalize to [0,1]
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr)
    # Fail if size is not 256×256 (user wants a hard assert, no silent fix)
    assert img.size == (256, 256), f"Expected image size (256, 256) but got {img.size}"

    out_name = f"{file_prefix}.png"
    img.save(os.path.join(out_dir, out_name))

# ---------------------------
# Process Train Cells
# ---------------------------
for d in train_cells:
    tifs = sorted(glob.glob(os.path.join(d, "TR1_*.tif")))
    if tifs:
        # e.g., "cellXYZ"
        cell_name = os.path.basename(os.path.dirname(d))

        # For each frame we want to extract
        for f_idx in FRAMES_TO_GRAB:
            if f_idx < len(tifs):
                # e.g., TR1_000.tif if f_idx=0
                tif_path = tifs[f_idx]
                # e.g., "cellXYZ_f0"
                file_prefix = f"{cell_name}_f{f_idx}"
                process_tif(tif_path, TRAIN_DIR, file_prefix)

# ---------------------------
# Process Test Cells
# ---------------------------
for d in test_cells:
    tifs = sorted(glob.glob(os.path.join(d, "TR1_*.tif")))
    if tifs:
        cell_name = os.path.basename(os.path.dirname(d))
        for f_idx in FRAMES_TO_GRAB:
            if f_idx < len(tifs):
                tif_path = tifs[f_idx]
                file_prefix = f"{cell_name}_f{f_idx}"
                process_tif(tif_path, TEST_DIR, file_prefix)
