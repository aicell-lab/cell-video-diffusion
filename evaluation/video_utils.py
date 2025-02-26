"""Utility functions for loading, processing and saving fluorescence microscopy videos.

This module provides functions for:
- Loading and saving video files
- Fluorescence image enhancement using Fourier filtering
- Contrast enhancement and normalization
- Creating overlays with segmentation masks
"""

import os
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


def load_video(video_path: str) -> np.ndarray:
    """Load video file and convert frames to grayscale.

    Parameters
    ----------
    video_path : str
        Path to the video file

    Returns
    -------
    np.ndarray
        Grayscale frames array with shape (T, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Load frames with progress bar
    saved = False
    for _ in tqdm(range(total_frames), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        if not saved:
            # Save the first frame for preview
            cv2.imwrite("preview.png", frame)
            saved = True

    cap.release()
    frames = np.stack(frames)
    print(f"Loaded {len(frames)} frames with shape {frames.shape}")

    return frames


def fourier_high_pass(image: np.ndarray, ksize: int = 3, cutoff: int = 3) -> np.ndarray:
    """Apply high-pass filter in Fourier domain.

    Removes low-frequency background noise while preserving high-frequency details.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, shape (H, W), dtype uint8
    ksize : int, optional
        Kernel size for Gaussian blur, by default 3
    cutoff : int, optional
        Radius for low-frequency suppression, by default 5

    Returns
    -------
    np.ndarray
        Filtered image, shape (H, W), dtype uint8
    """
    # Smoothing to prevent artifacts
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # Convert to float and Fourier transform
    f = np.fft.fft2(blurred)
    fshift = np.fft.fftshift(f)

    # Create a high-pass mask (inverse Gaussian)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= cutoff**2
    mask[mask_area] = 0  # Remove low-frequency components

    # Apply mask and inverse transform
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image
    clip_limit : float, optional
        Threshold for contrast limiting, by default 2.0
    tile_grid_size : int, optional
        Size of grid for histogram equalization, by default 8

    Returns
    -------
    np.ndarray
        CLAHE enhanced image
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
    )
    image_clahe = clahe.apply(image)
    return image_clahe


def percentile_normalization(
    image: np.ndarray, percentile: Tuple[float, float] = (0.5, 99.5)
) -> np.ndarray:
    """Normalize image using sigmoid-based percentile scaling.

    Parameters
    ----------
    image : np.ndarray
        Input image, dtype float32
    percentile : Tuple[float, float], optional
        Lower and upper percentiles, by default (0.5, 99.5)

    Returns
    -------
    np.ndarray
        Normalized image, dtype uint8
    """
    image = image.astype(np.float32)
    p_low, p_high = np.percentile(image, percentile)
    image = np.clip(image, p_low, p_high)
    scaled = (image - p_low) / (p_high - p_low)

    # Sigmoid stretch for smooth transition to preserve fine details
    # scaled = 1 / (1 + np.exp(-10 * (scaled - 0.5)))  # [0, 1]

    return scaled


def enhance_fluorescence(
    image: np.ndarray,
    ksize: int = 11,
    cutoff: int = 3,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    percentile: Tuple[float, float] = (0.01, 99.9),
) -> np.ndarray:
    """Enhance fluorescence microscopy images.

    Applies sequential filtering operations:
    1. Fourier high-pass to remove background
    2. CLAHE for local contrast enhancement
    3. Soft percentile normalization

    Parameters
    ----------
    image : np.ndarray
        Input grayscale fluorescence image with shape (H, W)
    ksize : int, optional
        Kernel size for high-pass filter, by default 11
    cutoff : int, optional
        Cutoff frequency for high-pass, by default 3
    clip_limit : float, optional
        CLAHE clip limit, by default 2.0
    tile_grid_size : int, optional
        CLAHE tile size, by default 8
    percentile : Tuple[float, float], optional
        Normalization percentiles, by default (0.01, 99.9)

    Returns
    -------
    np.ndarray
        Enhanced image, dtype float32 with values in [0, 1]
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must have shape (H, W)")
    if image.dtype != np.uint8:
        raise ValueError("Input image must be dtype uint8")

    # Step 1: Remove Low-Frequency Background (Fourier High-Pass)
    # high_pass_filtered = fourier_high_pass(image, ksize=ksize, cutoff=cutoff)
    high_pass_filtered = image

    # Step 2: Apply CLAHE for local contrast enhancement
    image_clahe = apply_clahe(
        high_pass_filtered, clip_limit=clip_limit, tile_grid_size=tile_grid_size
    )

    frame_blur = cv2.GaussianBlur(image_clahe, (11, 11), 0)

    # Step 3: Soft Percentile Normalization
    scaled = percentile_normalization(frame_blur, percentile=percentile)

    return scaled


def preprocess_video(
    frames: np.ndarray,
    preview_path: str = None,
    **enhance_kwargs,
) -> np.ndarray:
    """Enhance fluorescence microscopy video frames.

    Applies the `enhance_fluorescence` function to each frame.

    Parameters
    ----------
    frames : np.ndarray
        Input video frames with shape (T, H, W)
    preview_path : str, optional
        Path to save the first frame comparison image, by default None
    enhance_kwargs : dict, optional
        Keyword arguments for `enhance_fluorescence`

    Returns
    -------
    np.ndarray
        Enhanced video frames with shape (T, H, W)
    """
    enhanced_frames = []
    for i, frame in enumerate(tqdm(frames, desc="Enhancing frames")):
        enhanced_frame = enhance_fluorescence(frame, **enhance_kwargs)
        enhanced_frames.append(enhanced_frame)
        if preview_path is not None and i == 0:
            comparison_image = np.hstack(
                (frame, (enhanced_frame * 255).astype(np.uint8))
            )
            cv2.imwrite(preview_path, comparison_image)
            file_name = os.path.basename(preview_path)
            print(f"\nPreview saved to {file_name}")
            print("Press Enter to continue processing remaining frames...")
            input()

    enhanced_frames = np.stack(enhanced_frames)
    print(f"Enhanced frames with shape {enhanced_frames.shape}")
    return enhanced_frames


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.8,
) -> np.ndarray:
    """Create overlay of a single image with its segmentation mask.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape (H, W)
    mask : np.ndarray
        Segmentation mask with shape (H, W)
    color : Tuple[int, int, int], optional
        BGR color for overlay, by default (0, 0, 255)
    alpha : float, optional
        Alpha blending factor, by default 0.8

    Returns
    -------
    np.ndarray
        Overlay image with shape (H, W, 3)
    """
    if image.dtype != np.uint8:
        raise ValueError("Frame must be dtype uint8")
    # Convert grayscale to BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Create blend for mask areas
    mask_rgb = np.zeros_like(image_bgr)
    mask_rgb[mask > 0] = color
    # Blend original image with mask using specified alpha
    overlay = cv2.addWeighted(image_bgr, alpha, mask_rgb, 1 - alpha, 0)
    return overlay


def save_overlay(
    frame: np.ndarray, 
    mask: np.ndarray,
    overlay: np.ndarray,
    out_path: str,
):
    """Save a preview image showing original frame, segmentation mask and overlay.

    Parameters
    ----------
    frame : np.ndarray
        Original grayscale frame, shape (H, W), dtype uint8
    mask : np.ndarray
        Cellpose segmentation mask, shape (H, W), dtype uint16/int32
    overlay : np.ndarray
        Pre-computed overlay image, shape (H, W, 3)
    out_path : str
        Path to save the preview image
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Ensure frame is uint8 and in BGR format for display
    if frame.dtype != np.uint8:
        raise ValueError("Frame must be dtype uint8")
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Create visualization of cellpose mask
    mask_viz = np.zeros_like(frame)
    if mask.max() > 0:  # Avoid division by zero
        mask_viz = ((mask > 0) * 255).astype(np.uint8)
    mask_viz_bgr = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
    
    # Stack images horizontally: [Original | Mask | Overlay]
    comparison = np.hstack([frame_bgr, mask_viz_bgr, overlay])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)  # White text
    
    h, w = frame.shape
    labels = ['Original', 'Mask', 'Overlay']
    for i, label in enumerate(labels):
        # Position text above each image in the comparison
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x = (i * w + (w - text_size[0]) // 2)  # Center text
        y = 30  # Y position from top
        cv2.putText(comparison, label, (x, y), font, font_scale, color, thickness)
    
    # Save the comparison image
    cv2.imwrite(out_path, comparison)

    file_name = os.path.basename(out_path)
    print(f"Overlay preview saved to {file_name}")


def create_video_overlay(
    frames: np.ndarray,
    masks: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.8,
    preview_path: str = None,
) -> np.ndarray:
    """Create overlay video with segmentation masks for all frames.

    Parameters
    ----------
    frames : np.ndarray
        Frames array with shape (T, H, W) and dtype uint8
    masks : np.ndarray
        Masks array with shape (T, H, W)
    color : Tuple[int, int, int], optional
        BGR color for overlay, by default (0, 0, 255)
    alpha : float, optional
        Alpha blending factor, by default 0.8
    preview_path : str, optional
        Path to save the first frame comparison image, by default None

    Returns
    -------
    np.ndarray
        Overlay frames array with shape (T, H, W, 3)
    """
    if frames.shape != masks.shape:
        raise ValueError("Frames and masks must have the same shape")
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)
    overlay_frames = []
    for i, (frame, mask) in enumerate(tqdm(zip(frames, masks), desc="Creating overlays", total=len(frames))):
        overlay = create_overlay(frame, mask, color, alpha)
        overlay_frames.append(overlay)
        if preview_path is not None and i == 0:
            save_overlay(frame, mask, overlay, preview_path)
            print("Press Enter to continue processing remaining frames...")
            input()

    overlay_frames = np.stack(overlay_frames)
    print(f"Created overlay frames with shape {overlay_frames.shape}")
    return overlay_frames


def save_video(frames: np.ndarray, save_path: str, fps: int = 30):
    """
    Save frames as video file

    Parameters
    ----------
    frames : np.ndarray
        Frames array with shape (T, H, W, 3)
    save_path : str
        Path to save the video file
    fps : int, optional
        Frames per second, by default 30
    """
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h), isColor=True)

    # Write frames with progress bar
    for frame in tqdm(frames, desc="Saving frames"):
        out.write(frame)

    out.release()
    file_name = os.path.basename(save_path)
    print(f"Video saved to {file_name}")


if __name__ == "__main__":
    # Example usage
    samples = (
        "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/LT0001_02/00001_01.mp4",
    )
    video_path = samples[0]
    sample_name = os.path.splitext(os.path.basename(video_path))[0]
    preview_dir = os.path.join(os.path.dirname(__file__), "preview")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_path = os.path.join(
        preview_dir, f"enhanced_{sample_name}_{timestamp}.png"
    )

    # Load video
    frames = load_video(video_path)
    print("\nVideo loaded successfully!")

    # Process video frames
    enhanced_image = preprocess_video(
        frames,
        preview_path=preview_path,
        ksize=11,
        cutoff=3,
        clip_limit=2.0,
        tile_grid_size=8,
        percentile=(30, 99.9),
    )
