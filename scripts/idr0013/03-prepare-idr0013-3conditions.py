"""
prepare_idr0013.py

Example usage:
  python prepare_idr0013.py \
    --csv /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/idr0013-screenA-annotation.csv \
    --data_root /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013 \
    --output_dir ../../data/ready/IDR0013-VidGene

Creates:
  ./IDR0013-VidGene/prompts.txt + videos.txt  (training)
  ./IDR0013-VidGene-Val/prompts.txt + videos.txt (validation)
Where each prompt has binned text for migration speed/distance/proliferation,
based on percentiles from the dataset distribution.
"""

import csv
import os
import argparse
import re
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to idr0013-screenA-annotation.csv")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing plate folders (e.g. LT0001_02, LT0001_09, ...)")
    parser.add_argument("--output_dir", type=str, default="./IDR0013-VidGene-Binned",
                        help="Base output directory name (we also create `...-Val` for validation).")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Limit total matched samples. -1 means no limit.")
    parser.add_argument("--val_samples", type=int, default=1,
                        help="How many samples to reserve for validation. Default=1.")
    parser.add_argument("--percentiles", type=str, default="33,66",
                        help="Comma-separated percentile boundaries for binning (e.g. '33,66').")
    args = parser.parse_args()

    train_dir = args.output_dir
    val_dir   = f"{args.output_dir}-Val"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Parse percentile boundaries
    # e.g. "33,66" -> [33.0, 66.0]
    pcts = [float(x) for x in args.percentiles.split(",")]
    pcts.sort()  # ensure ascending

    # We'll read numeric columns from CSV, store them to compute distribution.
    # Then we define bins based on those percentile values.

    annotation_dict = {}
    plateWell_map   = {}

    # Step 1: collect numeric columns
    mig_speed_vals = []
    mig_dist_vals  = []
    prolif_vals    = []

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate_well = row["Plate_Well"]
            plate      = row["Plate"]
            well_num   = row["Well Number"]

            def safe_float(x):
                try:
                    return float(x)
                except:
                    return 0.0

            ms = safe_float(row.get("Score - migration (speed) (automatic)", "0"))
            md = safe_float(row.get("Score - migration (distance) (automatic)", "0"))
            pr = safe_float(row.get("Score - increased proliferation (automatic)", "0"))

            annotation_dict[plate_well] = {
                "plate": plate,
                "well_num": well_num,
                "mig_speed": ms,
                "mig_dist":  md,
                "prolif":    pr
            }

            mig_speed_vals.append(ms)
            mig_dist_vals.append(md)
            prolif_vals.append(pr)

    # Step 2: build (plate, well_num) -> plate_well map
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate      = row["Plate"]
            well_num   = row["Well Number"]
            plate_well = row["Plate_Well"]
            plateWell_map[(plate, well_num)] = plate_well

    # Step 3: compute numeric percentile boundaries
    # We'll define a helper that returns e.g. [lowBound, highBound] for 2 bin edges
    # if user passes "33,66", we get e.g. 2 edges => 3 bins => [lowest, 1stEdge, 2ndEdge, upTo max]
    # We can do up to n edges => n+1 bins.

    def get_bin_edges(values, percentile_list):
        arr = np.array(values)
        arr = arr[~np.isnan(arr)]  # remove any NaNs
        edges = []
        for p in percentile_list:
            # get percentile p
            ed = np.percentile(arr, p)
            edges.append(ed)
        return edges

    # For example, if pcts=[33,66], we have 2 edges => 3 bins
    edges_mig_speed = get_bin_edges(mig_speed_vals, pcts)  # e.g. [x1, x2]
    edges_mig_dist  = get_bin_edges(mig_dist_vals, pcts)
    edges_prolif    = get_bin_edges(prolif_vals, pcts)

    # We'll define a function that given x, edges=[e1,e2], labels=[LOW,MED,HIGH]
    # returns the label
    # e.g. if x < e1 => 'LOW'
    # else if x < e2 => 'MED'
    # else => 'HIGH'

    def bin_value(x, edges, labels):
        # e.g. edges=[e1, e2], labels=[LOW, MED, HIGH]
        # Must have len(labels)==len(edges)+1
        for i, e in enumerate(edges):
            if x < e:
                return labels[i]
        return labels[-1]

    # We'll define label sets for the 3 columns
    # If we have only 2 edges => 3 bins
    # You can do more edges => more bins
    # E.g. edges=[25,50,75] => 4 bins => labels=[A,B,C,D]
    n_edges = len(pcts)
    n_bins = n_edges + 1

    # We'll define the label sets. If n_bins=3 => [LOW,MED,HIGH]
    # If n_bins=4 => [BIN1,BIN2,BIN3,BIN4], etc.
    # We'll do a general approach
    def make_labels(n_bins, base_str):
        """
        Returns a list of label strings for the number of bins n_bins.
        Example: if n_bins=3, we return ["low", "medium", "high"].
        """
        if n_bins == 3:
            return ["low", "medium", "high"]  # Instead of ["LOW","MED","HIGH"]
        elif n_bins == 4:
            return ["very_low", "low", "medium", "high"]  # Example if you want 4 bins
        else:
            # fallback
            return [f"{base_str}_{i}" for i in range(n_bins)]

    labels_mig_speed = make_labels(n_bins, "MS")
    labels_mig_dist  = make_labels(n_bins, "MD")
    labels_prolif    = make_labels(n_bins, "PR")

    # Step 4: gather final matched prompts
    prompts = []
    videopaths = []

    # We define a function for binning the row's numeric
    def bin_mig_speed(x):
        return bin_value(x, edges_mig_speed, labels_mig_speed)

    def bin_mig_dist(x):
        return bin_value(x, edges_mig_dist, labels_mig_dist)

    def bin_prolif(x):
        return bin_value(x, edges_prolif, labels_prolif)

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

            data = annotation_dict.get(plate_well)
            if not data:
                continue

            ms = data["mig_speed"]
            md = data["mig_dist"]
            pr = data["prolif"]

            ms_bin = bin_mig_speed(ms)
            md_bin = bin_mig_dist(md)
            pr_bin = bin_prolif(pr)

            # e.g. "Time-lapse microscopy video. Migration speed=LOW, distance=MED, proliferation=HIGH."
            prompt_text = (
                f"Time-lapse microscopy video with {ms_bin} migration speed, "
                f"{md_bin} migration distance, and {pr_bin} proliferation."
            )

            abs_video_path = os.path.abspath(os.path.join(plate_path, mp4file))

            prompts.append(prompt_text)
            videopaths.append(abs_video_path)

    total_found = len(prompts)
    print(f"Found {total_found} matched videos in total.")

    # Limit if needed
    if args.max_samples > 0 and args.max_samples < total_found:
        prompts   = prompts[:args.max_samples]
        videopaths = videopaths[:args.max_samples]
        total_found = len(prompts)

    # Split off val_samples
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