#!/usr/bin/env python3

"""
02-score-videos.py

Scans plate folders under a given data root (e.g. /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013),
for each .mp4 file:
  - Extracts well number from the filename (e.g. "00005_01.mp4" => well_num=5)
  - Computes proliferation ratio = (last_frame_count / first_frame_count)
  - Writes results to a CSV with columns:
      Plate, Well Number, Proliferation Score, Video Path

Usage Example:
  python 02-score-videos.py \
    --data_root /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val \
    --output_csv ./idr0013_scores_val.csv
"""

import argparse
import csv
import re
import cv2
import numpy as np
from pathlib import Path

def segment_and_label(frame_bgr, threshold=50, min_area=5):
    """
    Threshold + connected components segmentation.
    Returns a 2D 'labels' array for each pixel (0=background).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(bin_img)

    for label_id in range(1, num_labels):
        area = np.sum(labels == label_id)
        if area < min_area:
            labels[labels == label_id] = 0

    return labels

def count_nuclei(frame_bgr, threshold=50, min_area=5):
    """
    Returns the number of distinct connected components > 0.
    """
    labels = segment_and_label(frame_bgr, threshold=threshold, min_area=min_area)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]  # skip background (label 0)
    return len(unique_ids)

def compute_proliferation_ratio(video_path, threshold=50, min_area=5):
    """
    Returns ratio = (last_frame_nuclei / first_frame_nuclei) or 0 if first_frame_nuclei=0.
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        cap.release()
        return 0.0

    # Read the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0
    first_count = count_nuclei(first_frame, threshold=threshold, min_area=min_area)

    # Read the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, last_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0
    last_count = count_nuclei(last_frame, threshold=threshold, min_area=min_area)

    cap.release()

    ratio = (last_count / first_count) if first_count > 0 else 0.0
    return ratio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder containing plate subdirs with .mp4 (e.g. /proj/aicell/.../idr0013)")
    parser.add_argument("--output_csv", type=str, default="./idr0013_prolif_only.csv",
                        help="Output CSV file storing (Plate, WellNum, Proliferation Score, VideoPath)")
    parser.add_argument("--threshold", type=int, default=50,
                        help="Threshold for segmentation (default=50)")
    parser.add_argument("--min_area", type=int, default=5,
                        help="Minimum area for connected components (default=5)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"Error: data_root={data_root} is not a directory.")
        return

    # ------------------------------------------------------
    # 1) Check if output CSV exists and load existing rows.
    # ------------------------------------------------------
    outpath = Path(args.output_csv)
    existing_entries = set()
    file_exists = outpath.is_file()

    if file_exists:
        with outpath.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Mark existing Plate & Well as known
                existing_entries.add((row["Plate"], row["Well Number"]))

    plate_folders = [p for p in data_root.iterdir() if p.is_dir()]
    output_rows = []

    # ------------------------------------------------------
    # 2) Only compute if (Plate, WellNum) missing from CSV.
    # ------------------------------------------------------
    for plate_dir in plate_folders:
        plate_name = plate_dir.name  # e.g. "LT0001_02"
        mp4_files = list(plate_dir.glob("*.mp4"))

        for mp4path in mp4_files:
            match = re.match(r"^0*(\d+)_\d+\.mp4$", mp4path.name)
            if not match:
                continue
            well_num_str = match.group(1)  # e.g. "5"

            # If already in CSV, skip
            if (plate_name, well_num_str) in existing_entries:
                continue

            prolifer = compute_proliferation_ratio(mp4path,
                                                   threshold=args.threshold,
                                                   min_area=args.min_area)

            output_rows.append({
                "Plate": plate_name,
                "Well Number": well_num_str,
                "Proliferation Score": f"{prolifer:.4f}",
                "Video Path": str(mp4path)
            })

    # ------------------------------------------------------
    # 3) Append (or write) new rows to CSV
    # ------------------------------------------------------
    # If the file doesn't exist, write header. Otherwise append.
    write_header = not file_exists

    with outpath.open("a" if file_exists else "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Plate", "Well Number", "Proliferation Score", "Video Path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for row in output_rows:
            writer.writerow(row)

    print(f"Skipped existing entries. Wrote {len(output_rows)} new rows to {outpath}")

if __name__ == "__main__":
    main()