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
    --output_dir ../../data/ready/IDR0013-10plates \
    --percentiles 33,66 \
    --val_samples 1 \
    --val_path /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/LT0001_02/00223_01.mp4
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
                        help="Comma-separated percentile boundaries for binning.")
    parser.add_argument("--val_samples", type=int, default=1,
                        help="Number of videos to move into validation set. Default=1.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="If >0, limit total number of samples.")
    # New param for optional single validation path
    parser.add_argument("--val_path", type=str, default=None,
                        help="If set and val_samples=1, force this video path into validation set.")
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

    # 5) Train/val split with optional forced val_path
    #    By default, we reserve the last val_samples entries for validation.
    #    But if val_samples=1 and val_path is specified, we force that video into validation.
    valN = min(args.val_samples, total_found)
    trainN = total_found - valN

    # If they provided --val_path but val_samples != 1, ignore it with a warning:
    if args.val_path and args.val_samples != 1:
        print("Warning: --val_path was set, but val_samples != 1. Ignoring val_path.")

    # Initialize with default approach:
    train_prompts = prompts[:trainN]
    train_videos  = videos[:trainN]
    val_prompts   = prompts[trainN:]
    val_videos    = videos[trainN:]

    # If we are indeed dealing with val_samples=1 and a val_path is provided:
    if valN == 1 and args.val_path:
        if args.val_path in videos:
            # Find the index of the provided path
            idx = videos.index(args.val_path)

            # Force that single sample to the validation set
            val_prompts = [prompts[idx]]
            val_videos  = [videos[idx]]

            # Rebuild the training set (all except that one sample)
            train_prompts = prompts[:idx] + prompts[idx+1:]
            train_videos  = videos[:idx]  + videos[idx+1:]
            print(f"Forcing {args.val_path} into validation set.")
        else:
            print(f"Warning: --val_path {args.val_path} not found in data. Using default last-item val.")

    print(f"Training samples: {len(train_prompts)}, Validation samples: {len(val_prompts)}")

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

    print(f"Wrote training set => {len(train_prompts)} lines:\n  {out_prompts_train}\n  {out_videos_train}")
    print(f"Wrote validation set => {len(val_prompts)} lines:\n  {out_prompts_val}\n  {out_videos_val}")

if __name__ == "__main__":
    main()