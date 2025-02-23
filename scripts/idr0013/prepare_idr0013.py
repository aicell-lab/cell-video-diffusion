#!/usr/bin/env python3

"""
prepare_idr0013.py

Example usage:
  python prepare_idr0013.py \
    --csv /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/idr0013-screenA-annotation.csv \
    --data_root /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013 \
    --output_dir ../../data/ready/IDR0013-VidGene

What it does:
  1) Collect matching video–prompt pairs from your dataset.
  2) Keep only up to `max_samples`.
  3) Reserve the last `val_samples` lines for validation.
  4) Writes:
       ./IDR0013-VidGene/prompts.txt
       ./IDR0013-VidGene/videos.txt
     for training
  5) Also writes:
       ./IDR0013-VidGene-Val/prompts.txt
       ./IDR0013-VidGene-Val/videos.txt
     for validation.
"""

import csv
import os
import argparse
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to idr0013-screenA-annotation.csv")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing plate folders (e.g. LT0001_02, LT0001_09, ...)")
    parser.add_argument("--output_dir", type=str, default="./IDR0013-VidGene",
                        help="Base output directory name (we also create `...-Val` for validation).")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Limit total matched samples. -1 means no limit.")
    parser.add_argument("--val_samples", type=int, default=1,
                        help="How many samples to reserve for validation. Default=1.")
    args = parser.parse_args()

    # We'll produce 2 sets of files:
    #   {output_dir}/prompts.txt + videos.txt      <-- training set
    #   {output_dir}-Val/prompts.txt + videos.txt  <-- validation set
    #
    # The script is the same as before, but at the end we split the matched data.

    # For convenience, define final output paths:
    train_dir = args.output_dir
    val_dir   = f"{args.output_dir}-Val"

    # Make sure the train directory exists, and the val directory as well.
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # We'll create two lookups:
    # 1) annotation_dict: Key = Plate_Well, storing numeric scores
    # 2) plateWell_map:   Key = (plate, well_number) -> Plate_Well
    annotation_dict = {}
    plateWell_map = {}

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate_well = row["Plate_Well"]   # e.g. "LT0001_02_A1"
            plate      = row["Plate"]        # e.g. "LT0001_02"
            well_num   = row["Well Number"]  # e.g. "1" or "384"

            # read numeric scores (3 columns)
            def safe_float(x):
                try:
                    return float(x)
                except:
                    return 0.0

            mig_speed = safe_float(row.get("Score - migration (speed) (automatic)", "0"))
            mig_dist  = safe_float(row.get("Score - migration (distance) (automatic)", "0"))
            prolif    = safe_float(row.get("Score - increased proliferation (automatic)", "0"))

            annotation_dict[plate_well] = {
                "plate": plate,
                "well_num": well_num,
                "mig_speed": mig_speed,
                "mig_dist":  mig_dist,
                "prolif":    prolif
            }

    # Build the plateWell_map in a second pass (or same pass) for convenience
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate      = row["Plate"]
            well_num   = row["Well Number"]
            plate_well = row["Plate_Well"]
            plateWell_map[(plate, well_num)] = plate_well

    prompts = []
    videopaths = []

    # Walk each plate folder
    for plate_folder in os.listdir(args.data_root):
        plate_path = os.path.join(args.data_root, plate_folder)
        if not os.path.isdir(plate_path):
            continue

        mp4s = [f for f in os.listdir(plate_path) if f.endswith(".mp4")]
        for mp4file in mp4s:
            match = re.match(r"^0*(\d+)_\d+\.mp4$", mp4file)
            if not match:
                continue

            well_num_str = match.group(1)
            plate_well = plateWell_map.get((plate_folder, well_num_str))
            if not plate_well:
                continue

            data = annotation_dict.get(plate_well, None)
            if not data:
                continue

            mig_speed = data["mig_speed"]
            mig_dist  = data["mig_dist"]
            prolif    = data["prolif"]

            # Very simple prompt using raw numeric values
            prompt_text = (
                f"Time-lapse microscopy video. "
                f"Migration speed={mig_speed:.3f}, distance={mig_dist:.3f}, proliferation={prolif:.3f}."
            )

            abs_video_path = os.path.abspath(os.path.join(plate_path, mp4file))

            prompts.append(prompt_text)
            videopaths.append(abs_video_path)

    total_found = len(prompts)
    print(f"Found {total_found} matched videos in total.")

    # 1) Limit with max_samples if >0
    if args.max_samples > 0 and args.max_samples < total_found:
        prompts   = prompts[:args.max_samples]
        videopaths = videopaths[:args.max_samples]
        total_found = len(prompts)

    # 2) Separate the last `val_samples` for validation
    valN = min(args.val_samples, total_found)
    trainN = total_found - valN

    train_prompts = prompts[:trainN]
    train_videos  = videopaths[:trainN]
    val_prompts   = prompts[trainN:]
    val_videos    = videopaths[trainN:]

    print(f"Training samples: {len(train_prompts)}  Validation samples: {len(val_prompts)}")

    # Write training set
    out_prompts_train = os.path.join(train_dir, "prompts.txt")
    out_videos_train  = os.path.join(train_dir, "videos.txt")
    with open(out_prompts_train, "w", encoding="utf-8") as f_p, open(out_videos_train, "w", encoding="utf-8") as f_v:
        for p, v in zip(train_prompts, train_videos):
            f_p.write(p + "\n")
            f_v.write(v + "\n")

    # Write validation set
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