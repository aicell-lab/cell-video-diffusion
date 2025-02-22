#!/usr/bin/env python3

"""
verify_videos_with_firstframe.py

Usage:
  python verify_videos_with_firstframe.py --videos_txt videos.txt

This script:
1. Reads each .mp4 path from `videos.txt`.
2. Tries to open with OpenCV.
3. Reads the first frame -> ensures it's valid (not empty).
4. (Optional) simulates a color conversion or minimal resize step, to ensure the frame is truly usable for i2v logic.

Any file that fails is printed as an error.
"""

import argparse
import os
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_txt", type=str, required=True, help="Path to videos.txt")
    args = parser.parse_args()

    with open(args.videos_txt, "r", encoding="utf-8") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    print(f"Verifying {len(video_paths)} videos by reading their first frame...")
    bad_count = 0

    for i, vid_path in enumerate(video_paths, start=1):
        if i % 50 == 0:
            print(f"Checked {i} videos so far...")

        if not os.path.isfile(vid_path):
            print(f"[MISSING] File does not exist: {vid_path}")
            bad_count += 1
            continue

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {vid_path}")
            bad_count += 1
            cap.release()
            continue

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(f"[ERROR] Cannot read first frame: {vid_path}")
            bad_count += 1
            continue

        # (Optional) Convert to RGB, just like typical code (like i2v) might do:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_rgb is None or frame_rgb.size == 0:
            print(f"[ERROR] Empty RGB conversion for: {vid_path}")
            bad_count += 1
            continue

        # (Optional) If you want to test a small resize:
        # e.g. to see if something about your final shape is problematic
        # height, width = 200, 200
        # resized = cv2.resize(frame_rgb, (width, height))
        # if resized is None or resized.size == 0:
        #    print(f"[ERROR] Resize step failed for: {vid_path}")
        #    bad_count += 1
        #    continue

    print(f"\nDone. Found {bad_count} problematic videos out of {len(video_paths)}.")

if __name__ == "__main__":
    main()
