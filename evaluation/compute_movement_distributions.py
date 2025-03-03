#!/usr/bin/env python3

"""
compute_movement_distributions.py

Given a folder of .npy mask files (one per video),
this script:

1) Loads each (T, H, W) .npy mask file.
2) Tracks nuclei across frames in each video.
3) Extracts movement metrics (speed, distance, directness) for each tracked nucleus.
4) Aggregates all metrics into a single distribution across all videos.
5) Saves:
   - A CSV file containing all per-nucleus movement metrics: all_movement.csv
   - A 2x2 histogram plot (avg_speed, max_speed, total_distance, directness).

Usage Example:
  python compute_movement_distributions.py \
      --input_dir masks_output/i2v_r128_250

Dependencies:
  - movement_analysis.py (track_nuclei, compute_movement_metrics)
  - skimage, pandas, matplotlib, numpy, tqdm
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import functions from movement_analysis.py
from movement_analysis import track_nuclei, compute_movement_metrics


def compute_movement_distributions(input_dir, output_dir):
    """
    1) Iterates over all .npy mask files in input_dir.
    2) Loads each (T,H,W) mask array.
    3) Tracks nuclei and computes movement metrics for each video.
    4) Aggregates data into a DataFrame, saves CSV & histogram plots.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Collect movement metrics for ALL videos
    all_tracks = []  # Will store DataFrames of track metrics for each video

    # Get list of .npy files
    mask_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    print(f"Found {len(mask_files)} .npy mask file(s) in {input_dir}")

    for idx, mask_name in enumerate(mask_files, start=1):
        mask_path = os.path.join(input_dir, mask_name)
        print(f"\n[{idx}/{len(mask_files)}] Processing {mask_name} ...")

        # 1) Load masks (shape = (T, H, W))
        masks = np.load(mask_path)
        if masks.ndim != 3:
            print(f"Warning: {mask_name} does not have shape (T,H,W). Skipping...")
            continue
        
        # 2) Track nuclei in the video
        print(f"Tracking nuclei in {mask_name}...")
        tracks = track_nuclei(masks)
        
        # 3) Compute movement metrics for the tracks
        print(f"Computing movement metrics for {len(tracks)} tracks...")
        metrics_df = compute_movement_metrics(tracks)
        
        # Add video name to identify source
        metrics_df['video'] = mask_name
        
        all_tracks.append(metrics_df)

    # Combine all tracks into a single DataFrame
    if all_tracks:
        df_movement = pd.concat(all_tracks, ignore_index=True)
        print(f"\nCollected metrics for {len(df_movement)} tracked nuclei across all videos.")
    else:
        print("No valid tracks found in any videos!")
        return

    # 4) Save raw data
    csv_path = os.path.join(output_dir, "all_movement.csv")
    df_movement.to_csv(csv_path, index=False)
    print(f"Saved all metrics to {csv_path}")

    if df_movement.empty:
        print("No tracks found in any videos! Exiting without plotting.")
        return

    # 5) Plot histograms for the 4 main metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ["avg_speed", "max_speed", "total_distance", "directness"]
    titles = ["Average Speed (pixels/frame)", "Maximum Speed (pixels/frame)", 
              "Total Distance (pixels)", "Directness Ratio"]

    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        vals = df_movement[metric].dropna()
        ax.hist(vals, bins=50, alpha=0.7, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "movement_histograms.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Histogram figure saved to {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute aggregated movement distributions (speed, distance, directness) from precomputed .npy masks."
    )
    parser.add_argument("--input_dir", required=True, help="Folder with .npy mask files.")
    parser.add_argument("--output_dir", help="Where to save CSV & histograms. If not provided, will automatically use 'results/X' where X is the model identifier from input_dir.")
    args = parser.parse_args()

    # Auto-determine output directory if not provided
    output_dir = args.output_dir
    if output_dir is None:
        # Extract the model identifier part from the input path
        if '/' in args.input_dir:
            # If input is like "masks_output/i2v_r128_250"
            parts = args.input_dir.split('/')
            model_id = parts[-1]  # Get the last part (i2v_r128_250)
            output_dir = os.path.join("results", model_id)
        else:
            # Just in case the input doesn't have directories
            output_dir = os.path.join("results", args.input_dir)
        
        print(f"Auto-determined output directory: {output_dir}")

    compute_movement_distributions(
        input_dir=args.input_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main() 