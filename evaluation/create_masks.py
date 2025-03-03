#!/usr/bin/env python3

"""
create_masks.py

1) For each .mp4 either:
   - From a directory of .mp4 files (input_dir)
   - OR from a text file with one video path per line (input_file)
   Then:
   - Load & preprocess frames
   - Segment with Cellpose
   - Save the resulting masks as .npy

2) Output is placed in output_dir, 
   e.g. "my_video_001.mp4" -> "my_video_001_masks.npy"

Then you can reuse these .npy mask files in your morphology or division analyses.

Usage:
  # From directory:
  python create_masks.py --input_dir /path/to/videos --output_dir /path/to/masks
  
  # From text file:
  python create_masks.py --input_file /path/to/videos.txt --output_dir /path/to/masks
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

# local modules
from video_utils import load_video, preprocess_video
from segmentation import segment_video

def convert_videos_to_masks(videos, output_dir, diameter=None, flow_threshold=0.6, cellprob_threshold=-2, min_size=15):
    """
    Convert a list of video paths to mask arrays.
    
    Args:
        videos: Either a list of video paths or a directory containing videos
        output_dir: Directory to save mask files
        other parameters: Passed to segment_video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, video_path in enumerate(videos, start=1):
        video_name = os.path.basename(video_path)
        
        # Check if mask already exists
        base_name = os.path.splitext(video_name)[0]
        mask_filename = f"{base_name}_masks.npy"
        mask_path = os.path.join(output_dir, mask_filename)
        
        if os.path.exists(mask_path):
            print(f"\n[{i}/{len(videos)}] Skipping {video_name} - mask already exists at {mask_path}")
            continue
            
        print(f"\n[{i}/{len(videos)}] Processing {video_path}")
        
        # 1) Load frames
        frames_uint8 = load_video(video_path)
        
        # 2) Preprocess frames
        frames_float = preprocess_video(frames_uint8)
        
        # 3) Segment
        masks = segment_video(
            frames_float,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            preview_path=None
        )
        
        # 4) Save .npy
        np.save(mask_path, masks)
        print(f"Saved masks to {mask_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert .mp4 videos to .npy Cellpose masks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", help="Directory containing .mp4 videos.")
    group.add_argument("--input_file", help="Text file with one video path per line.")
    parser.add_argument("--output_dir", required=True, help="Where to save .npy mask files.")
    parser.add_argument("--diameter", type=int, default=None, help="Cellpose approximate diameter.")
    parser.add_argument("--flow_threshold", type=float, default=0.6, help="Cellpose flow threshold.")
    parser.add_argument("--cellprob_threshold", type=float, default=-2, help="Cellpose cellprob threshold.")
    parser.add_argument("--min_size", type=int, default=15, help="Minimum nucleus size in pixels.")
    args = parser.parse_args()

    # Get list of video paths either from directory or from file
    if args.input_dir:
        video_files = sorted(os.path.join(args.input_dir, f) 
                     for f in os.listdir(args.input_dir) if f.endswith(".mp4"))
        print(f"Found {len(video_files)} .mp4 files in {args.input_dir}")
    else:
        with open(args.input_file, 'r') as f:
            video_files = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(video_files)} video paths from {args.input_file}")

    convert_videos_to_masks(
        videos=video_files,
        output_dir=args.output_dir,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size
    )

if __name__ == "__main__":
    main()