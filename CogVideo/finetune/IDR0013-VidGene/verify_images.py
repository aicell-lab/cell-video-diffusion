#!/usr/bin/env python3

"""
verify_images.py

Usage:
    python verify_images.py --images_dir /path/to/first_frames

This script:
1. Reads every file with .png in a given directory.
2. Attempts to load the image with OpenCV (cv2.imread).
3. Logs any files that fail.
"""

import argparse
import os
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Directory containing .png first-frame images")
    args = parser.parse_args()

    image_files = [f for f in os.listdir(args.images_dir) if f.endswith(".png")]
    bad_count = 0

    for i, filename in enumerate(image_files, start=1):
        if i % 50 == 0:
            print(f"Checked {i} images so far...")

        path = os.path.join(args.images_dir, filename)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[ERROR] Corrupt or unreadable PNG: {path}")
            bad_count += 1

    print(f"\nDone. Found {bad_count} problematic images out of {len(image_files)}.")

if __name__ == "__main__":
    main()
