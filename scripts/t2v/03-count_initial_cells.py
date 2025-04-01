#!/usr/bin/env python3

"""
03-count_initial_cells.py

Counts the number of cells in the first frame of each video and adds
'initial_cell_count' and 'initial_cell_count_label' columns to the CSV.

Usage:
  python 03-count_initial_cells.py \
    --input_csv ./output/extreme_phenotypes_with_videos.csv \
    --output_csv ./output/extreme_phenotypes_with_cell_counts.csv \
    --preview_every 50  # Save preview image for every 50th video
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
from cellpose import models
from tqdm import tqdm
from skimage.measure import label
from scipy import ndimage

INPUT_CSV_PATH = "/proj/aicell/users/x_aleho/video-diffusion/scripts/t2v/output/extreme_phenotypes_with_videos.csv"
OUTPUT_CSV_PATH = "/proj/aicell/users/x_aleho/video-diffusion/scripts/t2v/output/extreme_phenotypes_with_cell_counts.csv"

def load_first_frame(video_path):
    """Load only the first frame of a video file.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        np.ndarray: First frame as grayscale image
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read video: {video_path}")
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return frame

def preprocess_frame(frame, use_clahe=False):
    """Enhance a single fluorescence microscopy frame for better segmentation.
    
    Args:
        frame (np.ndarray): Input frame with shape (H, W)
        use_clahe (bool): Whether to apply CLAHE enhancement
    
    Returns:
        np.ndarray: Enhanced frame normalized to [0, 1]
    """
    # Apply Gaussian blur with larger kernel to reduce noise
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    
    if use_clahe:
        # Apply CLAHE with reduced clip limit
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
    else:
        # Skip CLAHE
        enhanced = blurred
    
    # Normalize to [0, 1] using percentile-based normalization
    p_low, p_high = np.percentile(enhanced, (1, 99))
    normalized = (enhanced.astype(np.float32) - p_low) / (p_high - p_low)
    normalized = np.clip(normalized, 0, 1)
    
    return normalized

def segment_frame(frame, diameter=30):
    """Run Cellpose segmentation on a single frame.
    
    Args:
        frame (np.ndarray): Input frame as numpy array (H, W) with values in [0, 1]
        diameter (int): Approximate nuclei size
    
    Returns:
        np.ndarray: Segmentation mask
    """
    if frame.dtype != np.float32:
        frame = frame.astype(np.float32)
    
    # Initialize Cellpose model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.Cellpose(
        model_type="nuclei", gpu=torch.cuda.is_available(), device=device
    )
    
    # Run Cellpose
    mask, _, _, _ = model.eval(
        frame,
        channels=[0, 0],
        diameter=diameter,
        flow_threshold=0.7,
        cellprob_threshold=-1.0,
        min_size=25,
    )
    
    return mask

def count_nuclei(segmentation_mask):
    """Count the number of nuclei in a segmentation mask.
    
    Args:
        segmentation_mask (np.ndarray): Binary mask of shape (H, W)
        
    Returns:
        int: Number of nuclei (connected components)
    """
    labeled_mask = label(segmentation_mask)
    return labeled_mask.max()  # The maximum label value is the number of objects

def create_preview_image(frame, mask, count):
    """Create a preview image with original frame, mask, and overlay.
    
    Args:
        frame (np.ndarray): Original preprocessed frame [0,1]
        mask (np.ndarray): Segmentation mask
        count (int): Cell count to display
        
    Returns:
        np.ndarray: Preview image in BGR format
    """
    # Convert frame to uint8 for visualization
    frame_uint8 = (frame * 255).astype(np.uint8)
    
    # Create BGR versions
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
    
    # Create mask visualization
    mask_viz = np.zeros_like(frame)
    if mask.max() > 0:
        mask_viz = ((mask > 0) * 255).astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
    
    # Create overlay
    overlay = frame_bgr.copy()
    overlay[mask > 0] = [0, 0, 255]  # Red overlay
    
    # Stack images horizontally
    preview = np.hstack([frame_bgr, mask_bgr, overlay])
    
    # Add count text
    cv2.putText(preview, f"Cell count: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return preview

def label_cell_count(count, percentiles):
    """Label cell count as HIGH, MED, or LOW based on percentiles.
    
    Args:
        count (int): Cell count
        percentiles (dict): Dictionary with 'low' and 'high' percentile values
    
    Returns:
        str: "HIGH", "MED", or "LOW"
    """
    if count <= percentiles['low']:
        return "LOW"
    elif count >= percentiles['high']:
        return "HIGH"
    else:
        return "MED"

def main():
    parser = argparse.ArgumentParser(description="Count initial cells in videos")
    parser.add_argument('--input_csv', type=str, default=INPUT_CSV_PATH,
                        help='Path to input CSV with video paths')
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV_PATH,
                        help='Path to output CSV with cell counts')
    parser.add_argument('--diameter', type=int, default=30,
                        help='Approximate cell diameter for Cellpose')
    parser.add_argument('--low_percentile', type=float, default=10.0,
                        help='Percentile threshold for LOW cell count')
    parser.add_argument('--high_percentile', type=float, default=90.0,
                        help='Percentile threshold for HIGH cell count')
    parser.add_argument('--preview', action='store_true',
                        help='Save preview images of segmentation for all videos')
    parser.add_argument('--preview_dir', type=str, default='./output/cell_count_previews',
                        help='Directory to save preview images')
    parser.add_argument('--preview_every', type=int, default=0,
                        help='Save preview image for every N-th video (0 = disabled)')
    parser.add_argument('--use_clahe', action='store_true',
                        help='Apply CLAHE preprocessing (may enhance artifacts)')
    args = parser.parse_args()
    
    
    # Load the CSV with video information
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        return
    
    print(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples")
    
    # Create preview directory if needed
    if args.preview or args.preview_every > 0:
        os.makedirs(args.preview_dir, exist_ok=True)
        print(f"Preview images will be saved to {args.preview_dir}")
        
        if args.preview_every > 0:
            print(f"Saving preview image for every {args.preview_every}th video")
    
    # Process each video and count cells
    print("\nCounting cells in videos...")
    cell_counts = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_path = row['video_path']
        
        try:
            # Load first frame
            frame = load_first_frame(video_path)
            
            # Preprocess frame
            enhanced_frame = preprocess_frame(frame, use_clahe=args.use_clahe)
            
            # Segment frame
            mask = segment_frame(enhanced_frame, diameter=args.diameter)
            
            # Count nuclei
            count = count_nuclei(mask)
            cell_counts.append(count)
            
            # Determine if we should save a preview for this video
            save_preview = args.preview or (args.preview_every > 0 and i % args.preview_every == 0)
            
            # Save preview if requested
            if save_preview:
                # Create visualization
                preview = create_preview_image(enhanced_frame, mask, count)
                
                # Save preview
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                preview_path = os.path.join(args.preview_dir, f"{video_name}_count_{count}.png")
                cv2.imwrite(preview_path, preview)
                
                if args.preview_every > 0 and i % args.preview_every == 0:
                    print(f"\nSaved preview #{i//args.preview_every + 1}: {os.path.basename(preview_path)}")
                
        except Exception as e:
            print(f"\nError processing {video_path}: {str(e)}")
            cell_counts.append(None)  # Use None for failed videos
    
    # Add cell counts to DataFrame
    df['initial_cell_count'] = cell_counts
    
    # Calculate percentiles for labeling
    valid_counts = [c for c in cell_counts if c is not None]
    percentiles = {
        'low': np.percentile(valid_counts, args.low_percentile),
        'high': np.percentile(valid_counts, args.high_percentile)
    }
    print(f"\nCell count percentiles: {percentiles['low']} (low) and {percentiles['high']} (high)")
    
    # Add labels based on percentiles
    df['initial_cell_count_label'] = df['initial_cell_count'].apply(
        lambda x: label_cell_count(x, percentiles) if x is not None else None
    )
    
    # Save the enhanced CSV
    print(f"\nSaving enhanced data to {args.output_csv}")
    df.to_csv(args.output_csv, index=False)
    
    # Print statistics
    print("\nCell count statistics:")
    print(f"Min: {df['initial_cell_count'].min()}")
    print(f"Max: {df['initial_cell_count'].max()}")
    print(f"Mean: {df['initial_cell_count'].mean():.2f}")
    print(f"Median: {df['initial_cell_count'].median()}")
    
    # Print label distribution
    label_counts = df['initial_cell_count_label'].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} videos ({count/len(df)*100:.1f}%)")
    
    print(f"\nComplete! Processed {len(df)} videos")

if __name__ == "__main__":
    main() 