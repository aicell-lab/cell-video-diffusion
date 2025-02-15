import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class MCAFrameDataset(Dataset):
    def __init__(self, image_dir, prompt="a cell", transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.prompt = prompt
        self.transform = transform

    def get_dataset_info(self):
        return {
            "size": len(self.image_paths),
            "directory": os.path.dirname(self.image_paths[0]) if self.image_paths else None,
            "prompt": self.prompt
        }        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("L")  # Grayscale
        
        # Convert single channel -> 3-channel by replicate
        image = image.convert("RGB")  # This replicates L into R=G=B internally

        if self.transform:
            image = self.transform(image)

        return {
            "pixel_values": image,  # [3, 256, 256] after transforms
            "text": self.prompt     # or any dummy prompt
        }
    
class MCAFrameDatasetSingleChannel(Dataset):
    """
    For Approach B:
      - We load the PNG as single-channel ("L").
      - We do NOT replicate to 3 channels.
      - We'll rely on the model's VAE being modified to accept in_channels=1.
    """
    def __init__(self, image_dir, prompt="a cell", transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.prompt = prompt
        self.transform = transform

    def get_dataset_info(self):
        return {
            "size": len(self.image_paths),
            "directory": os.path.dirname(self.image_paths[0]) if self.image_paths else None,
            "prompt": self.prompt
        }        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Open as grayscale
        image = Image.open(path).convert("L")  # shape [H, W] single channel

        if self.transform:
            # After transforms.ToTensor(), shape will be [1, H, W]
            image = self.transform(image)

        return {
            "pixel_values": image,  # [1, 256, 256] after transforms
            "text": self.prompt
        }
