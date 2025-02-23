#!/usr/bin/env python3

"""
prepare_idr0013.py

Usage:
  python prepare_idr0013.py \
    --csv /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/idr0013-screenA-annotation.csv \
    --data_root /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013 \
    --output_dir ./IDR0013-VidGene-50

This will produce:
  ./IDR0013-VidGene/prompts.txt
  ./IDR0013-VidGene/videos.txt

Both files have the same number of lines. The i-th line in prompts.txt
matches the i-th line in videos.txt.
"""

import csv
import os
import argparse
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to idr0013-screenA-annotation.csv")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing plate folders (e.g. LT0001_02, LT0001_09, ...)")
    parser.add_argument("--output_dir", type=str, default="./IDR0013-VidGene", help="Where to write prompts.txt/videos.txt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read CSV into a dict:
    # Key = Plate_Well (e.g. "LT0001_02_A1")
    # Value = (GeneSymbol, HasPhenotype, Possibly other info)
    annotation_dict = {}
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate_well = row["Plate_Well"]  # e.g. "LT0001_02_A1"
            gene_symbol = row.get("Gene Symbol", "unknown").strip()
            has_phenotype = row.get("Has Phenotype", "").strip().lower()
            # Build a short text to mention domain context
            # We can store a prompt template here:
            annotation_dict[plate_well] = {
                "plate": row["Plate"],              # e.g. "LT0001_02"
                "well": row["Well"],               # e.g. "A1"
                "gene_symbol": gene_symbol,
                "has_phenotype": has_phenotype if has_phenotype else "no",
            }

    # Next, we'll search each plate folder (like LT0001_02) for .mp4 files.
    # We'll try to match them to a Plate_Well by scanning the filename for well info.
    # But many times, well is something like "A1" => how does that appear in the mp4?
    #
    # Hypothesis: The user mentioned "00384_01.mp4" as an example. Possibly "001_01.mp4" = well #1?
    # If so, we can parse out the well number from the filename and combine with the plate to form "LT0001_02_A1".
    # We also have a column "Well Number" => row["Well Number"].

    # Let's load the CSV again but keep row["Well Number"] so we can attempt numeric matching.
    # Or we can do it in a single pass, but let's keep it simpler:

    # We'll build a dictionary from (Plate, wellNum) -> Plate_Well
    # Then for each .mp4 that we find in plate folder "Plate", parse the wellNum from the filename and match.
    plateWell_map = {}  # (plate, well_number) -> Plate_Well
    # We'll store the entire row so we can build a prompt
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate = row["Plate"]  # e.g. "LT0001_02"
            well_num = row["Well Number"]  # e.g. "1" => int
            plate_well = row["Plate_Well"] # e.g. "LT0001_02_A1"
            plateWell_map[(plate, well_num)] = plate_well

    prompts = []
    videopaths = []

    # Now let's walk each plate folder
    for plate_folder in os.listdir(args.data_root):
        plate_path = os.path.join(args.data_root, plate_folder)
        if not os.path.isdir(plate_path):
            continue
        # e.g. "LT0001_02"
        # We'll search for .mp4 in that folder
        # For each .mp4, we attempt to parse a well_num from the filename, e.g. "00384_01.mp4" => well_num=384
        # We'll do a simple regex that looks for something like ^0*(\d+)_\d+.mp4
        mp4s = [f for f in os.listdir(plate_path) if f.endswith(".mp4")]
        for mp4file in mp4s:
            # e.g. "00384_01.mp4"
            match = re.match(r"^0*(\d+)_\d+\.mp4$", mp4file)
            if not match:
                # If it doesn't match, skip or handle differently
                continue
            well_num_str = match.group(1)   # e.g. "384"
            # We'll see if (plate_folder, well_num_str) is in plateWell_map
            plate_key = plate_folder
            well_key = well_num_str

            # The CSV might store well as "384", so we must match. 
            # But note that row["Well Number"] might be "384" or "384.0" if it's read as float, etc. 
            # We'll just compare strings for now:
            # We'll do well_key_str = well_num_str => "384"
            # so we check plateWell_map.get((plate_folder, "384"))
            plate_well = plateWell_map.get((plate_key, well_key))
            if not plate_well:
                # No direct match in dictionary
                continue

            # Ok we found a match => we can build the prompt
            data = annotation_dict.get(plate_well, None)
            if not data:
                continue

            # Construct a domain prompt
            # Example:
            # "Time-lapse microscopy video of HeLa cells in LT0001_02_A1, with siRNA knockdown of INCENP, 
            #  showing noticeable cell division abnormalities."
            gene_symbol = data["gene_symbol"]
            has_phenotype = data["has_phenotype"]
            well_label = plate_well
            # Simple text:
            prompt_text = (
                f"Time-lapse microscopy video of HeLa cells in {well_label}, with siRNA knockdown of {gene_symbol}. "
                f"Fluorescently labeled chromosomes are observed. Phenotype: {has_phenotype}."
            )

            abs_video_path = os.path.join(plate_path, mp4file)
            abs_video_path = os.path.abspath(abs_video_path)

            prompts.append(prompt_text)
            videopaths.append(abs_video_path)

    print(f"Found {len(prompts)} matched videos in total.")

    # Take only the first 50 samples
    prompts = prompts[:50]
    videopaths = videopaths[:50]

    # Write out prompts.txt and videos.txt
    out_prompts = os.path.join(args.output_dir, "prompts.txt")
    out_videos  = os.path.join(args.output_dir, "videos.txt")
    with open(out_prompts, "w", encoding="utf-8") as f_p, open(out_videos, "w", encoding="utf-8") as f_v:
        for p, v in zip(prompts, videopaths):
            f_p.write(p + "\n")
            f_v.write(v + "\n")

    print(f"Wrote {len(prompts)} lines to:")
    print(f"  {out_prompts}")
    print(f"  {out_videos}")

if __name__ == "__main__":
    main()
