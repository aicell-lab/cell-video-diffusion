#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

def copy_videos_to_val_dir():
    # Define source file and destination directory
    source_file = "../data/ready/checkpoint-900-val/videos.txt"
    destination_dir = "../data/processed/idr0013/val/checkpoint-900-val"
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Read the source file
    with open(source_file, 'r') as f:
        video_paths = f.read().splitlines()
    
    # Copy each video to the destination directory
    for i, video_path in enumerate(video_paths):
        if not video_path.strip():  # Skip empty lines
            continue
            
        # Get the filename from the path
        filename = os.path.basename(video_path)
        
        # Define the destination path
        dest_path = os.path.join(destination_dir, filename)
        
        # Copy the file
        try:
            shutil.copy2(video_path, dest_path)
            print(f"[{i+1}/{len(video_paths)}] Copied: {filename}")
        except FileNotFoundError:
            print(f"[{i+1}/{len(video_paths)}] Error: File not found - {video_path}")
        except PermissionError:
            print(f"[{i+1}/{len(video_paths)}] Error: Permission denied - {video_path}")
        except Exception as e:
            print(f"[{i+1}/{len(video_paths)}] Error copying {video_path}: {str(e)}")

if __name__ == "__main__":
    print("Starting to copy videos to validation directory...")
    copy_videos_to_val_dir()
    print("Finished copying videos.")