import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from PIL import Image
import numpy as np
from torchvision.transforms.functional import resize


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")


##########  loaders  ##########


def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_images(image_path: Path) -> List[Path]:
    with open(image_path, "r", encoding="utf-8") as file:
        return [image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_images_from_videos(videos_path: List[Path]) -> List[Path]:
    """
    OBS: I just updated this function to include the plate name prefix (e.g., "LT0001_02-00001_01.png")
    to prevent naming conflicts when extracting frames from videos with identical names across different plates.
    """
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        # Get plate name from parent directory
        plate_name = video_path.parent.name
        # Create filename with plate prefix: "LT0001_02-00001_01.png"
        frame_path = first_frames_dir / f"{plate_name}-{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with plate prefix
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths


##########  preprocessors  ##########

def preprocess_image_with_resize(
    image_path: Path | str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single image using Pillow instead of OpenCV.
    Returns a torch.Tensor with shape [C, H, W] in RGB.

    Args:
        image_path: Path to the image file.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W], float32 in [0, 255].
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    path_str = image_path.as_posix()

    with Image.open(path_str) as pil_img:
        # Force to RGB
        pil_img = pil_img.convert("RGB")
        # Resize
        pil_img = pil_img.resize((width, height), Image.BICUBIC)
        # Convert to numpy float32 array
        np_img = np.array(pil_img, dtype=np.float32)  # shape (H, W, 3)
    
    # Convert (H, W, C) -> (C, H, W)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
    return tensor_img

# def preprocess_image_with_resize(
#     image_path: Path | str,
#     height: int,
#     width: int,
# ) -> torch.Tensor:
#     """
#     Loads and resizes a single image.

#     Args:
#         image_path: Path to the image file.
#         height: Target height for resizing.
#         width: Target width for resizing.

#     Returns:
#         torch.Tensor: Image tensor with shape [C, H, W] where:
#             C = number of channels (3 for RGB)
#             H = height
#             W = width
#     """
#     import pdb; pdb.set_trace()
#     if isinstance(image_path, str):
#         image_path = Path(image_path)
#     path = image_path.as_posix()
#     import os
#     size = os.path.getsize(path)
#     print(f"[DEBUG] Attempting to read {path}, size={size} bytes")
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (width, height))
#     image = torch.from_numpy(image).float()
#     image = image.permute(2, 0, 1).contiguous()
#     return image


def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)
    if video_num_frames < max_num_frames:
        # Get all frames first
        frames = video_reader.get_batch(list(range(video_num_frames)))
        # Repeat the last frame until we reach max_num_frames
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        return frames.float().permute(0, 3, 1, 2).contiguous()
    else:
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[:max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames


def preprocess_video_with_buckets(
    video_path: Path,
    resolution_buckets: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """
    Args:
        video_path: Path to the video file.
        resolution_buckets: List of tuples (num_frames, height, width) representing
            available resolution buckets.

    Returns:
        torch.Tensor: Video tensor with shape [F, C, H, W] where:
            F = number of frames
            C = number of channels (3 for RGB)
            H = height
            W = width

    The function processes the video through these steps:
        1. Finds nearest frame bucket <= video frame count
        2. Downsamples frames evenly to match bucket size
        3. Finds nearest resolution bucket based on dimensions
        4. Resizes frames to match bucket resolution
    """
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    resolution_buckets = [bucket for bucket in resolution_buckets if bucket[0] <= video_num_frames]
    if len(resolution_buckets) == 0:
        raise ValueError(f"video frame count in {video_path} is less than all frame buckets {resolution_buckets}")

    nearest_frame_bucket = min(
        resolution_buckets,
        key=lambda bucket: video_num_frames - bucket[0],
        default=1,
    )[0]
    frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))
    frames = video_reader.get_batch(frame_indices)
    frames = frames[:nearest_frame_bucket].float()
    frames = frames.permute(0, 3, 1, 2).contiguous()

    nearest_res = min(resolution_buckets, key=lambda x: abs(x[1] - frames.shape[2]) + abs(x[2] - frames.shape[3]))
    nearest_res = (nearest_res[1], nearest_res[2])
    frames = torch.stack([resize(f, nearest_res) for f in frames], dim=0)

    return frames


def load_phenotypes(phenotype_path: Path) -> List[torch.Tensor]:
    """
    Load phenotype data from a CSV file.
    
    Args:
        phenotype_path (Path): Path to the CSV file containing phenotype data
        
    Returns:
        List[torch.Tensor]: List of phenotype tensors, each of shape [num_phenotypes]
    """
    phenotypes = []
    with open(phenotype_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        # Skip header row
        for line in lines[1:]:
            line = line.strip()
            if len(line) > 0:
                # Parse comma-separated values into tensor
                values = [float(value.strip()) for value in line.split(",")]
                phenotypes.append(torch.tensor(values, dtype=torch.float32))
    return phenotypes
