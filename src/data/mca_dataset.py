import os
import glob
import re
import torch
from torch.utils.data import Dataset
from PIL import Image

class MCAFrameDataset(Dataset):
    def __init__(self, image_dir, base_prompt="a microscopy image of a cell", transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.base_prompt = base_prompt
        self.transform = transform

    def get_dataset_info(self):
        return {
            "size": len(self.image_paths),
            "directory": os.path.dirname(self.image_paths[0]) if self.image_paths else None,
            "prompt_format": self.base_prompt
        }        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)

        # 1) Parse frame index from filename; fail if no match
        match = re.search(r"_f(\d+)", filename)
        assert match, f"Filename '{filename}' does not contain '_f<number>'."
        frame_idx = int(match.group(1))

        image = Image.open(path).convert("L")

        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        prompt = f"{self.base_prompt} at frame {frame_idx}"

        return {
            "pixel_values": image,
            "text": prompt
        }
