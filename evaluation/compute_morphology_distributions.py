#!/usr/bin/env python3

"""
compute_morphology_distributions.py

Given a folder of .npy mask files (one per video),
this script:

1) Loads each (T, H, W) .npy mask file.
2) Extracts four morphology metrics (area, eccentricity, solidity, perimeter) 
   for each nucleus in every frame.
3) Aggregates all metrics into a single distribution across all videos.
4) Saves:
   - A CSV file containing all per-nucleus metrics: all_morphology.csv
   - A 2x2 histogram plot (area, eccentricity, solidity, perimeter).

Usage Example:
  python compute_morphology_distributions.py \
      --input_dir /path/to/npy_masks \
      --output_dir ./results/i2v_r128_250

Dependencies:
  - morphology_analysis.py (get_nucleus_morphology)
  - skimage, pandas, matplotlib, numpy, tqdm
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# We only need get_nucleus_morphology to compute shapes
from morphology_analysis import get_nucleus_morphology


def compute_morphology_distributions(input_dir, output_dir):
    """
    1) Iterates over all .npy mask files in input_dir.
    2) Loads each (T,H,W) mask array.
    3) For each frame, extracts morphological properties of each nucleus
       (area, eccentricity, solidity, perimeter).
    4) Aggregates data into a DataFrame, saves CSV & histogram plots.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Collect morphology metrics for ALL videos
    all_nuclei = []  # Will store dicts: {"video", "frame", "area", "eccentricity", "solidity", "perimeter"}

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
        
        T = masks.shape[0]

        # 2) Extract morphological properties from each frame
        for t in tqdm(range(T), desc=f"Frames in {mask_name}", leave=False):
            morph_props = get_nucleus_morphology(masks[t])
            # morph_props is a list of dicts: e.g. [{"area":..., "eccentricity":..., ...}, ...]
            for mp in morph_props:
                all_nuclei.append(
                    {
                        "video": mask_name,
                        "frame": t,
                        "area": mp["area"],
                        "eccentricity": mp["eccentricity"],
                        "solidity": mp["solidity"],
                        "perimeter": mp["perimeter"],
                    }
                )

    # Convert to DataFrame
    df_morph = pd.DataFrame(all_nuclei)
    print(f"\nCollected {len(df_morph)} nuclei across all mask files.")

    # 3) Save raw data
    csv_path = os.path.join(output_dir, "all_morphology.csv")
    df_morph.to_csv(csv_path, index=False)
    print(f"Saved all metrics to {csv_path}")

    if df_morph.empty:
        print("No nuclei found in any frames! Exiting without plotting.")
        return

    # 4) Plot histograms for the 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ["area", "eccentricity", "solidity", "perimeter"]

    for ax, metric in zip(axes.ravel(), metrics):
        vals = df_morph[metric].dropna()
        ax.hist(vals, bins=50, alpha=0.7, edgecolor="black")
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
        description="Compute aggregated morphology distributions (area, ecc., solidity, perimeter) from precomputed .npy masks."
    )
    parser.add_argument("--input_dir", required=True, help="Folder with .npy mask files.")
    parser.add_argument("--output_dir", required=True, help="Where to save CSV & histograms.")
    args = parser.parse_args()

    compute_morphology_distributions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()