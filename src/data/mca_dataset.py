import os
import glob
import re
from torch.utils.data import Dataset
from PIL import Image
import torch

def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])  # (B,3,H,W)
    frame_idxs = torch.tensor([ex["frame_idx"] for ex in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,  # (B,3,H,W)
        "frame_idx": frame_idxs        # (B,)
    }

class MCAFrameDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        image_dir: directory with images named something like ..._f19.png
        transform: torchvision transforms
        """
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.transform = transform

    def get_dataset_info(self):
        return {
            "size": len(self.image_paths),
            "directory": os.path.dirname(self.image_paths[0]) if self.image_paths else None,
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)

        # Parse frame index from filename, e.g. ..._f19.png
        match = re.search(r"_f(\d+)", filename)
        assert match, f"Filename '{filename}' must contain '_f<number>'."
        frame_idx = int(match.group(1))

        # Load and transform the image
        image = Image.open(path).convert("L")  # grayscale
        image = image.convert("RGB")          # expand to 3 channels

        if self.transform:
            image = self.transform(image)

        return {
            "pixel_values": image,     # (3,H,W) after transforms
            "frame_idx": frame_idx     # integer
        }
