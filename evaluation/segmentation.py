#%%
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
    diameter: int = None,  # Approximate nuclei size (e.g. 30)
    flow_threshold: float = 0.6,  # Slightly higher to catch dim nuclei
    cellprob_threshold: float = -2,  # Much lower to catch dim nuclei
    min_size: int = 15,  # Default value
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

    # Update defaults with any user-specified parameters
    default_params = {
        "diameter": diameter,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "min_size": min_size,
    }
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


#%%
if __name__ == "__main__":
    from video_utils import load_video, create_overlay, save_video

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Example usage
    preview_dir = os.path.join(os.path.dirname(__file__), "preview")
    data_dir = os.path.join(preview_dir, "data")
    segmentation_dir = os.path.join(preview_dir, "segmentation")
    os.makedirs(segmentation_dir, exist_ok=True)
    samples = sorted(os.listdir(data_dir))

    sample_idx = 0
    video_path = os.path.join(data_dir, samples[sample_idx])
    sample_name = os.path.splitext(samples[sample_idx])[0]
    print(f"Segmenting video: {sample_name}")

    # Load video
    frames = load_video(video_path)

    # Process video frames
    enhanced_image = preprocess_video(frames)

    # Segment video
    preview_path = os.path.join(segmentation_dir, f"segmented_{sample_name}_{timestamp}.png")
    masks = segment_video(enhanced_image, preview_path=preview_path)

    file_name = f"masks_{sample_name}_{timestamp}.npy"
    save_path = os.path.join(segmentation_dir, file_name)
    np.save(save_path, masks)
    print(f"Segmentation masks saved to {file_name}")

    # Create an overlay video
    overlay_frames = create_video_overlay(
        enhanced_image, masks, color=(0, 0, 255), alpha=0.8,
    )
    save_path = os.path.join(segmentation_dir, f"segmented_{sample_name}_{timestamp}.mp4")
    save_video(overlay_frames, save_path, fps=15)


# %%
