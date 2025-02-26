import os
from datetime import datetime

import numpy as np
import torch
from cellpose import models
from tqdm import tqdm

from video_utils import preprocess_video, create_overlay, save_overlay, create_video_overlay


def segment_video(
    frames: np.ndarray,
    preview_path: str = None,
    **cellpose_kwargs,
) -> np.ndarray:  # Changed return type to np.ndarray
    """
    Run Cellpose segmentation on a sequence of frames.

    Args:
        frames: Input frames as numpy array (T, H, W)
        preview: If True, saves first frame preview and waits for user input

    Returns:
        Numpy array of masks with shape (T, H, W)
    """
    if len(frames.shape) != 3:
        raise ValueError("Input frames must have shape (T, H, W)")
    if frames.dtype != np.float32:
        raise ValueError("Input frames must be dtype float32")
    if frames.min() < 0 or frames.max() > 1:
        raise ValueError("Input frames must be normalized to [0, 1]")

    # Initialize Cellpose model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Warning: Running on CPU, this might be slow!")
    model = models.Cellpose(
        model_type="nuclei", gpu=torch.cuda.is_available(), device=device
    )
    masks = []

    # Default Cellpose parameters that can be overridden
    default_params = {
        "diameter": 30,  # Approximate nuclei size
        "flow_threshold": 0.6,  # Slightly higher to catch dim nuclei
        "cellprob_threshold": -2,  # Much lower to catch dim nuclei
        "min_size": 15,  # Default value
        'stitch_threshold': -1,  # Disable stitching between frames
    }

    # Update defaults with any user-specified parameters
    cellpose_kwargs = {**default_params, **cellpose_kwargs}
    print(f"Running Cellpose with parameters: {cellpose_kwargs}")

    # Process remaining frames in batches
    for i, frame in enumerate(tqdm(frames, desc="Running Cellpose")):
        mask, _, _, _ = model.eval(
            frame,
            channels=[0, 0],
            channel_axis=None,  # Process as single 2D image
            invert=False,
            normalize=False,  # Already normalized
            do_3D=False,
            **cellpose_kwargs,
        )
        masks.append(mask)

        # Process first frame for preview if requested
        if preview_path is not None and i == 0:
            frame = (frame * 255).astype(np.uint8)
            overlay = create_overlay(frame, mask)
            save_overlay(frame, mask, overlay, preview_path)
            print("Press Enter to continue processing remaining frames...")
            input()

    # Stack masks to match frames dimensions
    masks = np.stack(masks)
    print(f"Final masks shape: {masks.shape}")
    assert masks.shape == frames.shape, "Masks shape doesn't match frames shape"

    return masks


if __name__ == "__main__":
    from video_utils import load_video, create_overlay, save_video

    samples = (
        "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/LT0001_02/00001_01.mp4",
        "/proj/aicell/users/x_aleho/video-diffusion/CogVideo/test_generations/i2v_eval1_night/LT0001_02-00223_01_noLORA_lowPROF.mp4",
        "/proj/aicell/users/x_aleho/video-diffusion/CogVideo/test_generations/i2v_eval2_night/LT0001_02-00223_01_S50_G8_F97_FPS16.mp4",
    )
    video_path = samples[0]

    sample_name = os.path.splitext(os.path.basename(video_path))[0]
    preview_dir = os.path.join(os.path.dirname(__file__), "preview")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load video
    frames = load_video(video_path)

    # Preprocess video frames
    preview_path = os.path.join(preview_dir, f"enhanced_{sample_name}_{timestamp}.png")
    frames_proc = preprocess_video(
        frames,
        preview_path=preview_path,
        ksize=3,
        cutoff=1,
        clip_limit=3.0,
        tile_grid_size=8,
        percentile=(10, 99.9),
    )

    # Segment video
    preview_path = os.path.join(preview_dir, f"segmented_{sample_name}_{timestamp}.png")
    masks = segment_video(frames_proc, preview_path=preview_path)

    file_name = f"masks_{sample_name}_{timestamp}.npy"
    save_path = os.path.join(preview_dir, file_name)
    np.save(save_path, masks)
    print(f"Segmentation masks saved to {file_name}")

    # Create an overlay video
    overlay_frames = create_video_overlay(
        frames_proc, masks, color=(0, 0, 255), alpha=0.8,
    )
    save_path = os.path.join(preview_dir, f"segmented_{sample_name}_{timestamp}.mp4")
    save_video(overlay_frames, save_path, fps=15)

