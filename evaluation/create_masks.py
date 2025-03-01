#!/usr/bin/env python3

"""
create_masks.py

1) For each .mp4 in input_dir:
   - Load & preprocess frames
   - Segment with Cellpose
   - Save the resulting masks as .npy

2) Output is placed in output_dir, 
   e.g. "my_video_001.mp4" -> "my_video_001_masks.npy"

Then you can reuse these .npy mask files in your morphology or division analyses.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

# local modules
from video_utils import load_video, preprocess_video
from segmentation import segment_video

def convert_videos_to_masks(input_dir, output_dir, diameter=None, flow_threshold=0.6, cellprob_threshold=-2, min_size=15):
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".mp4"))
    print(f"Found {len(video_files)} .mp4 files in {input_dir}")

    for i, video_name in enumerate(video_files, start=1):
        video_path = os.path.join(input_dir, video_name)
        
        # Check if mask already exists
        base_name = os.path.splitext(video_name)[0]
        mask_filename = f"{base_name}_masks.npy"
        mask_path = os.path.join(output_dir, mask_filename)
        
        if os.path.exists(mask_path):
            print(f"\n[{i}/{len(video_files)}] Skipping {video_name} - mask already exists at {mask_path}")
            continue
            
        print(f"\n[{i}/{len(video_files)}] Processing {video_name}")
        
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
    parser.add_argument("--input_dir", required=True, help="Directory containing .mp4 videos.")
    parser.add_argument("--output_dir", required=True, help="Where to save .npy mask files.")
    parser.add_argument("--diameter", type=int, default=None, help="Cellpose approximate diameter.")
    parser.add_argument("--flow_threshold", type=float, default=0.6, help="Cellpose flow threshold.")
    parser.add_argument("--cellprob_threshold", type=float, default=-2, help="Cellpose cellprob threshold.")
    parser.add_argument("--min_size", type=int, default=15, help="Minimum nucleus size in pixels.")
    args = parser.parse_args()

    convert_videos_to_masks(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size
    )

if __name__ == "__main__":
    main()