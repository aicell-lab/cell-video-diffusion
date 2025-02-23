"""
prepare_idr0013.py

Example usage:
  python prepare_idr0013.py \
    --csv /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/idr0013-screenA-annotation.csv \
    --data_root /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013 \
    --output_dir ../../CogVideo/finetune/IDR0013-VidGene-BIG

Outputs:
  ./IDR0013-VidGene/prompts.txt
  ./IDR0013-VidGene/videos.txt
Each line i in 'prompts.txt' corresponds to line i in 'videos.txt'.
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
                        help="Where to write prompts.txt/videos.txt")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Limit the total number of samples (for debugging). Use -1 for no limit.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # We'll create two lookups:
    # 1) annotation_dict: Key = Plate_Well (like "LT0001_02_A1"),
    #                     storing numeric scores for migration/proliferation, etc.
    # 2) plateWell_map:   Key = (plate, well_number) -> Plate_Well
    #                     so we can match .mp4 filenames to annotation.

    annotation_dict = {}
    plateWell_map = {}

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate_well = row["Plate_Well"]   # e.g. "LT0001_02_A1"
            plate      = row["Plate"]        # e.g. "LT0001_02"
            well_num   = row["Well Number"]  # e.g. "1" or "384"

            # read numeric scores (3 columns), default to 0.0 on errors/blank
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

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate      = row["Plate"]
            well_num   = row["Well Number"]
            plate_well = row["Plate_Well"]
            plateWell_map[(plate, well_num)] = plate_well

    prompts = []
    videopaths = []

    # Walk each plate folder under data_root
    for plate_folder in os.listdir(args.data_root):
        plate_path = os.path.join(args.data_root, plate_folder)
        if not os.path.isdir(plate_path):
            continue

        mp4s = [f for f in os.listdir(plate_path) if f.endswith(".mp4")]
        for mp4file in mp4s:
            # e.g. "00384_01.mp4"
            match = re.match(r"^0*(\d+)_\d+\.mp4$", mp4file)
            if not match:
                # skip any non-matching files
                continue

            well_num_str = match.group(1)   # "384"
            # Attempt to find plate_well
            plate_well = plateWell_map.get((plate_folder, well_num_str))
            if not plate_well:
                continue

            data = annotation_dict.get(plate_well, None)
            if not data:
                continue

            # Extract numeric scores
            mig_speed = data["mig_speed"]
            mig_dist  = data["mig_dist"]
            prolif    = data["prolif"]

            # Build a simple prompt
            # E.g.: "Time-lapse microscopy video. Migration speed=0.123, distance=4.567, proliferation=-0.003."
            prompt_text = (
                f"Time-lapse microscopy video of dividing cells. "
                f"Migration speed={mig_speed:.3f}, distance={mig_dist:.3f}, proliferation={prolif:.3f}."
            )

            abs_video_path = os.path.join(plate_path, mp4file)
            abs_video_path = os.path.abspath(abs_video_path)

            prompts.append(prompt_text)
            videopaths.append(abs_video_path)

    print(f"Found {len(prompts)} matched videos in total.")

    # Optionally limit
    if args.max_samples > 0:
        prompts = prompts[:args.max_samples]
        videopaths = videopaths[:args.max_samples]

    # Write out
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