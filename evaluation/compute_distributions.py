#!/usr/bin/env python3

"""
compute_distribution.py

Given a folder of .mp4 videos (real or generated),
this script:

1) Loads and preprocesses each video (background removal, CLAHE, normalization).
2) Segments each video with Cellpose (once per video).
3) Extracts four morphology metrics (area, eccentricity, solidity, perimeter) for each nucleus.
4) Aggregates all metrics into a single distribution across all videos.
5) Saves:
   - A CSV file containing all per-nucleus metrics.
   - A single 2x2 histogram plot (area, eccentricity, solidity, perimeter).

Usage Example:
  python compute_distribution.py \
      --input_dir /proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_plateval/i2v_r128_250 \
      --output_dir ./plots/i2v_r128_250

Dependencies:
  - segmentation.py (Cellpose pipeline)
  - video_utils.py (for load/preprocess)
  - morphology_analysis.py (get_nucleus_morphology)
  - cellpose, skimage, pandas, matplotlib, numpy, tqdm
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from video_utils import load_video, preprocess_video
from segmentation import segment_video
from morphology_analysis import get_nucleus_morphology


def compute_distributions(
    input_dir,
    output_dir,
    diameter=None,
    flow_threshold=0.6,
    cellprob_threshold=-2,
    min_size=15,
):
    """
    Processes each .mp4 in input_dir to compute area, eccentricity,
    solidity, and perimeter from all nuclei. Aggregates these into
    a single distribution, plots histograms, and saves CSV and a figure.

    Args:
        input_dir (str): Directory containing .mp4 files.
        output_dir (str): Output directory for histograms & CSV.
        diameter (int or None): Cellpose approximate diameter.
        flow_threshold (float): Cellpose flow threshold.
        cellprob_threshold (float): Cellpose cellprob threshold.
        min_size (int): Minimum nucleus size (Cellpose).

    Returns:
        None. (Saves artifacts to output_dir)
    """

    os.makedirs(output_dir, exist_ok=True)

    # Collect morphology metrics for ALL videos
    all_nuclei = []  # Will store dicts with {"area", "eccentricity", "solidity", "perimeter"}

    # Get list of .mp4 videos
    video_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mp4")])
    print(f"Found {len(video_files)} video(s) in {input_dir}")

    for idx, video_name in enumerate(video_files, start=1):
        video_path = os.path.join(input_dir, video_name)
        print(f"\n[{idx}/{len(video_files)}] Processing {video_name} ...")

        # 1) Load frames (grayscale, uint8)
        frames_uint8 = load_video(video_path)

        # 2) Preprocess frames -> float32 in [0,1]
        frames_float = preprocess_video(frames_uint8)

        # 3) Segment video with Cellpose
        masks = segment_video(
            frames_float,
            preview_path=None,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
        )

        # 4) Extract morph. metrics from each frame
        for t in range(masks.shape[0]):
            morph_props = get_nucleus_morphology(masks[t])
            # morph_props is a list of dicts: e.g. [{"area": ..., "eccentricity": ...}, ...]
            for mp in morph_props:
                all_nuclei.append(
                    {
                        "video": video_name,
                        "frame": t,
                        "area": mp["area"],
                        "eccentricity": mp["eccentricity"],
                        "solidity": mp["solidity"],
                        "perimeter": mp["perimeter"],
                    }
                )

    # Convert to DataFrame
    df_morph = pd.DataFrame(all_nuclei)
    print(f"\nCollected {len(df_morph)} nuclei across all videos.")

    # Save raw data
    csv_path = os.path.join(output_dir, "all_morphology.csv")
    df_morph.to_csv(csv_path, index=False)
    print(f"Saved all metrics to {csv_path}")

    if len(df_morph) == 0:
        print("No nuclei found in any frames! Exiting without plotting.")
        return

    # 5) Plot histograms for the 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ["area", "eccentricity", "solidity", "perimeter"]

    for ax, metric in zip(axes.ravel(), metrics):
        ax.hist(df_morph[metric], bins=50, color="blue", alpha=0.7, edgecolor="black")
        ax.set_title(f"{metric.capitalize()} Distribution")
        ax.set_xlabel(metric)
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "morphology_histograms.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Histogram figure saved to {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute aggregated morphology distributions (area, ecc., solidity, perimeter) for .mp4 videos."
    )
    parser.add_argument("--input_dir", required=True, help="Folder with .mp4 videos.")
    parser.add_argument("--output_dir", required=True, help="Where to save CSV & histograms.")
    parser.add_argument("--diameter", type=int, default=None, help="Cellpose approximate diameter.")
    parser.add_argument("--flow_threshold", type=float, default=0.6, help="Cellpose flow threshold.")
    parser.add_argument("--cellprob_threshold", type=float, default=-2, help="Cellpose cellprob threshold.")
    parser.add_argument("--min_size", type=int, default=15, help="Minimum size of nucleus for Cellpose.")
    args = parser.parse_args()

    compute_distributions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
    )


if __name__ == "__main__":
    main()