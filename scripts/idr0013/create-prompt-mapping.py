#!/usr/bin/env python3

"""
create-prompt-mapping.py

Creates a CSV file that maps from plate-well IDs to their corresponding prompts.
This script uses the same binning logic as 03-prepare-idr0013.py but outputs
a simple mapping CSV instead of separate prompt/video files.

Usage example:
  python create-prompt-mapping.py \
    --scores_csv ./scores_all.csv \
    --output_csv ./prompt_mapping_all.csv \
    --percentiles 33,66 \
    --prompt_prefix "<ALEXANDER>"
"""

import argparse
import csv
import os
import numpy as np
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", type=str, required=True,
                        help="CSV containing Plate, Well Number, Proliferation Score, Video Path.")
    parser.add_argument("--output_csv", type=str, default="./idr0013_prompt_mapping.csv",
                        help="Output CSV file mapping plate-well IDs to prompts.")
    parser.add_argument("--percentiles", type=str, default="33,66",
                        help="Comma-separated percentile boundaries for binning.")
    parser.add_argument("--prompt_prefix", type=str, default="<ALEXANDER>",
                        help="Prefix to add to the beginning of each prompt. Default=<ALEXANDER>")
    args = parser.parse_args()

    # 1) Load the CSV into memory
    data_rows = []
    prolif_scores = []

    with open(args.scores_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate  = row["Plate"]
            well   = row["Well Number"]
            pscore = row["Proliferation Score"]
            vpath  = row["Video Path"]

            # Extract plate-well ID from video path if available
            plate_well_id = None
            if vpath:
                # Try to extract plate-well ID from the video path
                # Assuming format like "LT0002_51-00268_01" somewhere in the path
                match = re.search(r'([A-Z0-9]+_\d+-\d+_\d+)', vpath)
                if match:
                    plate_well_id = match.group(1)
                else:
                    # Fallback: create ID from plate and well
                    # Ensure well number is 5 digits with leading zeros and add _01 suffix
                    try:
                        well_num = int(well)
                        plate_well_id = f"{plate}-{well_num:05d}_01"
                    except ValueError:
                        plate_well_id = f"{plate}-{well}_01"
            else:
                # If no video path, create ID from plate and well
                # Ensure well number is 5 digits with leading zeros and add _01 suffix
                try:
                    well_num = int(well)
                    plate_well_id = f"{plate}_{well_num:05d}_01"
                except ValueError:
                    plate_well_id = f"{plate}_{well}_01"

            # Convert to float
            try:
                pval = float(pscore)
            except:
                pval = 0.0

            data_rows.append({
                "plate": plate,
                "well": well,
                "plate_well_id": plate_well_id,
                "prolif_val": pval,
                "video_path": vpath
            })
            prolif_scores.append(pval)

    total_found = len(data_rows)
    print(f"Loaded {total_found} rows from {args.scores_csv}")

    if total_found == 0:
        print("No data found. Exiting.")
        return

    # 2) Parse percentile boundaries
    pcts = [float(x) for x in args.percentiles.split(",")]
    pcts.sort()  # ensure ascending

    # Get the bin edges
    def get_bin_edges(values, percentile_list):
        arr = np.array(values)
        edges = []
        for p in percentile_list:
            ed = np.percentile(arr, p)
            edges.append(ed)
        return edges

    edges_prolif = get_bin_edges([r["prolif_val"] for r in data_rows], pcts)

    # Define the label sets
    n_edges = len(pcts)
    n_bins = n_edges + 1

    if n_bins == 3:
        bin_labels = ["low", "medium", "high"]
    else:
        # fallback
        bin_labels = [f"bin{i+1}" for i in range(n_bins)]

    def bin_value(x, edges, labels):
        for i, edge in enumerate(edges):
            if x < edge:
                return labels[i]
        return labels[-1]

    # 3) Create mapping from plate-well IDs to prompts
    mapping_rows = []
    for row in data_rows:
        pval = row["prolif_val"]
        binned_p = bin_value(pval, edges_prolif, bin_labels)
        prompt_text = f"{args.prompt_prefix} Time-lapse microscopy video with {binned_p} proliferation."
        
        mapping_rows.append({
            "plate_well_id": row["plate_well_id"],
            "plate": row["plate"],
            "well": row["well"],
            "proliferation_score": row["prolif_val"],
            "proliferation_category": binned_p,
            "prompt": prompt_text,
            "video_path": row["video_path"]
        })

    # 4) Write out the mapping CSV
    with open(args.output_csv, "w", encoding="utf-8", newline='') as f:
        fieldnames = ["plate_well_id", "plate", "well", "proliferation_score", 
                     "proliferation_category", "prompt", "video_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow(row)

    print(f"Wrote mapping CSV with {len(mapping_rows)} entries to: {args.output_csv}")

if __name__ == "__main__":
    main() 