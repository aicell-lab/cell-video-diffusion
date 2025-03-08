#!/usr/bin/env python3

"""
plot_nuclei_over_time.py

Given a folder of .npy mask files, this script:

1) Loads each .npy mask array (T, H, W)
2) Computes nuclei count for each frame in each video
3) Creates a CSV with columns: [video_name, frame, nuclei_count]
4) Plots nuclei count vs. frame for all videos on a single chart

Usage Example:
  python plot_nuclei_over_time.py --input_dir masks_output/phenotype_pr_pr-HIGH
  
  This will create output in results/phenotype_pr_pr-HIGH automatically.
  
  You can also specify a custom output directory:
  python plot_nuclei_over_time.py --input_dir masks_output/x --output_dir custom/path
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from division_analysis import count_nuclei


def compute_nuclei_per_frame(masks):
    """Count nuclei in each frame of a video.
    
    Args:
        masks (np.ndarray): Binary masks of shape (T, H, W)
        
    Returns:
        np.ndarray: Array containing the number of nuclei in each frame
    """
    nuclei_counts = []
    
    for t in range(masks.shape[0]):
        count = count_nuclei(masks[t])
        nuclei_counts.append(count)
        
    return np.array(nuclei_counts)


def process_mask_files(input_dir, output_dir=None):
    """Process all mask files in the directory and create time series data.
    
    Args:
        input_dir (str): Directory containing .npy mask files
        output_dir (str, optional): Directory to save outputs. If None, 
                                   automatically derived from input_dir.
    """
    # If output_dir is not specified, derive it from input_dir
    if output_dir is None:
        # Check if input is in masks_output directory
        if 'masks_output' in input_dir:
            # Extract the subdirectory part
            subdir = input_dir.split('masks_output/')[-1]
            output_dir = os.path.join('results', subdir)
        else:
            # If input is not in masks_output, create a results subdir
            base_name = os.path.basename(input_dir)
            output_dir = os.path.join('results', base_name)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Find all .npy mask files
    mask_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    print(f"Found {len(mask_files)} .npy mask file(s) in {input_dir}")
    
    if not mask_files:
        print("No mask files found. Exiting.")
        return
    
    # Create a list to store all dataframes
    dfs = []
    
    # Process each mask file
    for mask_name in tqdm(mask_files, desc="Processing mask files"):
        mask_path = os.path.join(input_dir, mask_name)
        
        # Load masks
        masks = np.load(mask_path)
        if masks.ndim != 3:
            print(f"Warning: {mask_name} does not have shape (T,H,W). Skipping...")
            continue
        
        # Compute nuclei count for each frame
        counts = compute_nuclei_per_frame(masks)
        
        # Create a dataframe for this video
        video_df = pd.DataFrame({
            'video_name': mask_name,
            'frame': range(len(counts)),
            'nuclei_count': counts
        })
        
        dfs.append(video_df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "nuclei_counts_over_time.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved nuclei counts to {csv_path}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get unique video names
    video_names = combined_df['video_name'].unique()
    
    # Plot each video as a separate line
    for video_name in video_names:
        video_data = combined_df[combined_df['video_name'] == video_name]
        plt.plot(video_data['frame'], video_data['nuclei_count'], label=video_name)
    
    # If there are too many videos, don't show individual legends
    if len(video_names) > 10:
        plt.title(f"Nuclei Count Over Time ({len(video_names)} videos)")
    else:
        plt.title("Nuclei Count Over Time")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.xlabel("Frame")
    plt.ylabel("Number of Nuclei")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "nuclei_counts_over_time.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Count nuclei over time for each video and visualize."
    )
    parser.add_argument("--input_dir", required=True, help="Directory with .npy mask files.")
    parser.add_argument("--output_dir", required=False, help="Where to save CSV & plot. If not specified, automatically derived from input_dir.")
    args = parser.parse_args()
    
    process_mask_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main() 