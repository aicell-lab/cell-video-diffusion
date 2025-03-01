#!/usr/bin/env python3

"""
compute_proliferation_distribution.py

Given a folder of .npy mask files, this script:

1) Loads each .npy mask array (T, H, W).
2) Computes proliferation metrics:
   - initial_count
   - final_count
   - growth_ratio (final/initial)
   - growth_absolute (final - initial)
   - division_events_count (via detect_division_events)
   - avg_division_interval (if multiple events)
3) Aggregates these metrics across all .npy files into a single CSV.
4) Plots histograms of each metric.

Usage Example:
  python compute_proliferation_distribution.py --input_dir /path/to/masks --output_dir /path/to/output
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from division_analysis import (
    compute_nuclei_counts,
    detect_division_events
)


def compute_proliferation_metrics(masks: np.ndarray):
    """
    Given a 3D mask array (T, H, W), compute a dictionary of proliferation metrics:
      - initial_count
      - final_count
      - growth_ratio
      - growth_absolute
      - division_events_count
      - avg_division_interval (None if <2 events)
    """
    # 1) Count nuclei per frame
    counts = compute_nuclei_counts(masks)  # shape (T,)

    initial_count = counts[0]
    final_count   = counts[-1]

    # Avoid divide-by-zero
    if initial_count > 0:
        growth_ratio = final_count / initial_count
    else:
        growth_ratio = np.nan

    growth_abs = final_count - initial_count

    # 2) Detect division events
    division_frames = detect_division_events(counts)
    division_count  = len(division_frames)

    if division_count >= 2:
        avg_div_interval = np.mean(np.diff(division_frames))
    else:
        avg_div_interval = np.nan

    return {
        "initial_count": initial_count,
        "final_count": final_count,
        "growth_ratio": growth_ratio,
        "growth_absolute": growth_abs,
        "division_events_count": division_count,
        "avg_division_interval": avg_div_interval
    }


def compute_proliferation_distribution(input_dir, output_dir):
    """
    1) Iterates over all .npy files in input_dir.
    2) For each file, loads the mask array (T,H,W), calculates proliferation metrics.
    3) Aggregates these metrics into a CSV and a set of histograms.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gather all .npy mask files
    mask_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    print(f"Found {len(mask_files)} .npy mask file(s) in {input_dir}")

    all_stats = []
    for i, mask_name in enumerate(mask_files, start=1):
        mask_path = os.path.join(input_dir, mask_name)
        print(f"[{i}/{len(mask_files)}] Processing {mask_name} ...")

        # Load the (T,H,W) mask array
        masks = np.load(mask_path)
        if masks.ndim != 3:
            print(f"Warning: {mask_name} does not have shape (T,H,W). Skipping...")
            continue

        # Compute proliferation metrics
        stats = compute_proliferation_metrics(masks)
        stats["video"] = mask_name  # Keep track of which file
        all_stats.append(stats)

    # Convert to a DataFrame
    df = pd.DataFrame(all_stats)
    print(f"\nComputed proliferation metrics for {len(df)} videos.")

    # Save the raw data to CSV
    csv_path = os.path.join(output_dir, "proliferation_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved proliferation metrics to {csv_path}")

    if len(df) == 0:
        print("No valid .npy mask files processed. Exiting without plotting.")
        return

    # Define which metrics to plot
    metrics_to_plot = [
        "initial_count",
        "final_count",
        "growth_ratio",
        "growth_absolute",
        "division_events_count",
        "avg_division_interval"
    ]

    # Filter out columns not in the DataFrame (e.g. if missing)
    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    # Plot histograms
    num_metrics = len(metrics_to_plot)
    cols = 3
    rows = (num_metrics + cols - 1) // cols  # round up
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, metrics_to_plot):
        vals = df[metric].dropna()
        ax.hist(vals, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("Frequency")

    # If we have more axes than metrics, hide extra subplots
    for extra_ax in axes[num_metrics:]:
        extra_ax.set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "proliferation_histograms.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Histogram figure saved to {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot proliferation metrics distribution from .npy mask files."
    )
    parser.add_argument("--input_dir", required=True, help="Directory with .npy mask files.")
    parser.add_argument("--output_dir", required=True, help="Where to save CSV & histograms.")
    args = parser.parse_args()

    compute_proliferation_distribution(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()