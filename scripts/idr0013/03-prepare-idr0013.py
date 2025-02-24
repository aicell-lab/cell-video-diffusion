#!/usr/bin/env python3

"""
03-prepare_idr0013.py

Scans a CSV (e.g. 'idr0013_scores.csv') that contains:
    Plate, Well Number, Proliferation Score, Video Path
Then:
  1) Bins 'Proliferation Score' into e.g. LOW / MED / HIGH,
     using user-specified percentile boundaries (e.g. "33,66").
  2) Writes out two sets of matching lines: 
     - prompts.txt
     - videos.txt 
     for training and validation subsets.
  3) Prompts look like:
       "Time-lapse microscopy video with LOW proliferation."

Usage example:
  python 03-prepare-idr0013.py \
    --scores_csv ./idr0013_scores.csv \
    --output_dir ./IDR0013 \
    --percentiles 33,66 \
    --val_samples 1
"""

import argparse
import csv
import os
import re
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", type=str, required=True,
                        help="CSV containing Plate, Well Number, Proliferation Score, Video Path.")
    parser.add_argument("--output_dir", type=str, default="./IDR0013-Prolif-Binned",
                        help="Base output dir for train/validation subdirs.")
    parser.add_argument("--percentiles", type=str, default="33,66",
                        help="Comma-separated percentile boundaries for binning (e.g. '33,66').")
    parser.add_argument("--val_samples", type=int, default=1,
                        help="Number of videos to move into validation set. Default=1.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="If >0, limit total number of samples.")
    args = parser.parse_args()

    train_dir = args.output_dir
    val_dir   = f"{args.output_dir}-Val"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 1) Load the CSV into memory
    #    We'll store each row as a dict and also keep track of numeric proliferation scores
    data_rows = []
    prolif_scores = []

    with open(args.scores_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate  = row["Plate"]
            well   = row["Well Number"]
            pscore = row["Proliferation Score"]
            vpath  = row["Video Path"]

            # Convert to float
            try:
                pval = float(pscore)
            except:
                pval = 0.0

            data_rows.append({
                "plate": plate,
                "well":  well,
                "prolif_val": pval,
                "video_path": vpath
            })
            prolif_scores.append(pval)

    total_found = len(data_rows)
    print(f"Loaded {total_found} rows from {args.scores_csv}")

    if total_found == 0:
        print("No data found. Exiting.")
        return

    # 2) If max_samples specified, limit
    if args.max_samples > 0 and args.max_samples < total_found:
        data_rows   = data_rows[:args.max_samples]
        total_found = len(data_rows)
        print(f"Truncated to {total_found} samples due to max_samples={args.max_samples}")

    # 3) Parse percentile boundaries
    pcts = [float(x) for x in args.percentiles.split(",")]
    pcts.sort()  # ensure ascending

    # We'll define a function to get the actual bin edges
    def get_bin_edges(values, percentile_list):
        arr = np.array(values)
        edges = []
        for p in percentile_list:
            ed = np.percentile(arr, p)
            edges.append(ed)
        return edges

    edges_prolif = get_bin_edges([r["prolif_val"] for r in data_rows], pcts)
    # e.g. if pcts=[33,66], edges_prolif might be [0.72, 1.45], etc. => 3 bins => LOW, MED, HIGH

    # We'll define the label sets
    n_edges = len(pcts)  # e.g. 2
    n_bins  = n_edges + 1  # e.g. 3 => [LOW, MED, HIGH]

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

    # 4) Build final (prompt, video) pairs
    prompts, videos = [], []
    for row in data_rows:
        pval = row["prolif_val"]
        binned_p = bin_value(pval, edges_prolif, bin_labels)
        # e.g. "Time-lapse microscopy video with low proliferation."
        prompt_text = f"Time-lapse microscopy video with {binned_p} proliferation."
        prompts.append(prompt_text)
        videos.append(row["video_path"])

    # 5) Create train/val split
    valN = min(args.val_samples, total_found)
    trainN = total_found - valN

    train_prompts = prompts[:trainN]
    train_videos  = videos[:trainN]
    val_prompts   = prompts[trainN:]
    val_videos    = videos[trainN:]

    print(f"Training samples: {trainN}, Validation samples: {valN}")

    # 6) Write them out
    out_prompts_train = os.path.join(train_dir, "prompts.txt")
    out_videos_train  = os.path.join(train_dir, "videos.txt")
    with open(out_prompts_train, "w", encoding="utf-8") as f_p, open(out_videos_train, "w", encoding="utf-8") as f_v:
        for p, v in zip(train_prompts, train_videos):
            f_p.write(p + "\n")
            f_v.write(v + "\n")

    out_prompts_val = os.path.join(val_dir, "prompts.txt")
    out_videos_val  = os.path.join(val_dir, "videos.txt")
    with open(out_prompts_val, "w", encoding="utf-8") as f_p, open(out_videos_val, "w", encoding="utf-8") as f_v:
        for p, v in zip(val_prompts, val_videos):
            f_p.write(p + "\n")
            f_v.write(v + "\n")

    print(f"Wrote training set => {trainN} lines:\n  {out_prompts_train}\n  {out_videos_train}")
    print(f"Wrote validation set => {valN} lines:\n  {out_prompts_val}\n  {out_videos_val}")

if __name__ == "__main__":
    main()